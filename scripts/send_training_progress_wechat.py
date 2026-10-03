#!/usr/bin/env python3
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/ldl/mybot_v3_stair_turn_rl')
LOG = ROOT / 'logs' / 'training_latest.log'
PID_FILE = ROOT / 'logs' / 'training.pid'
STATE = Path('/home/ldl/.hermes/mybot-training-progress')
PENDING = STATE / 'pending.txt'
SEND_LOG = STATE / 'send.log'
LAST_STATUS = STATE / 'last_status.json'
LAST_SENT = STATE / 'last_sent.txt'
TARGET = 'weixin:o9cq808C_ry72g5NiXOcVDBWumFw@im.wechat'
TOTAL_ITERATIONS = 61000
PHASE_START_ITERATION = 59000
TRAINING_LABEL = 'sim2sim鲁棒第三阶段训练'
ACTIVE_RUN = None
ACTIVE_PID = None

# Discover the newest live mybot training process and its matching validation
# monitor. This keeps progress reporting correct when supervisors advance to a
# new phase or restart from a healthy checkpoint.
def _option(parts, name):
    try:
        return parts[parts.index(name) + 1]
    except (ValueError, IndexError):
        return None


active_trainers = []
for proc_dir in Path('/proc').iterdir():
    if not proc_dir.name.isdigit():
        continue
    try:
        parts = [part.decode(errors='replace') for part in
                 (proc_dir / 'cmdline').read_bytes().split(b'\0') if part]
        train_script = next((part for part in parts
                             if re.search(r'(?:^|/)scripts/train_mybot_v3_[^/]+\.py$', part)), None)
        if not train_script:
            continue
        start_ticks = int((proc_dir / 'stat').read_text().split()[21])
        active_trainers.append((start_ticks, int(proc_dir.name), parts, train_script))
    except (OSError, ValueError):
        continue

if active_trainers:
    _, ACTIVE_PID, active_parts, active_script = max(active_trainers)
    checkpoint_text = _option(active_parts, '--checkpoint')
    iterations_text = _option(active_parts, '--iterations')
    if checkpoint_text and iterations_text:
        PHASE_START_ITERATION = int(checkpoint_text)
        TOTAL_ITERATIONS = PHASE_START_ITERATION + int(iterations_text)

    for proc_dir in Path('/proc').iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            parts = [part.decode(errors='replace') for part in
                     (proc_dir / 'cmdline').read_bytes().split(b'\0') if part]
            monitor_pid = _option(parts, '--training-pid')
            run_text = _option(parts, '--run')
            if monitor_pid == str(ACTIVE_PID) and run_text:
                candidate = Path(run_text)
                if candidate.is_dir():
                    ACTIVE_RUN = candidate
                    break
        except OSError:
            continue

    if ACTIVE_RUN is None:
        pointers = sorted((ROOT / 'logs').glob('*_run_dir.txt'),
                          key=lambda path: path.stat().st_mtime, reverse=True)
        for pointer in pointers:
            try:
                candidate = Path(pointer.read_text().strip())
                if candidate.is_dir():
                    ACTIVE_RUN = candidate
                    break
            except OSError:
                continue

    if ACTIVE_RUN:
        LOG = ACTIVE_RUN / 'outputs.log'
    script_stem = Path(active_script).stem
    if 'phase4' in script_stem:
        TRAINING_LABEL = 'sim2sim动力学匹配第四阶段训练'
    elif 'phase3b' in script_stem:
        TRAINING_LABEL = 'sim2sim鲁棒第三阶段B训练'
    elif 'phase3' in script_stem:
        TRAINING_LABEL = 'sim2sim鲁棒第三阶段训练'
    else:
        TRAINING_LABEL = script_stem.replace('train_mybot_v3_', '')
else:
    # When the newest phase has just stopped, report that run instead of
    # falling back to an older completed phase.
    pointers = sorted((ROOT / 'logs').glob('*_run_dir.txt'),
                      key=lambda path: path.stat().st_mtime, reverse=True)
    for pointer in pointers:
        try:
            candidate = Path(pointer.read_text().strip())
            if candidate.is_dir():
                ACTIVE_RUN = candidate
                LOG = ACTIVE_RUN / 'outputs.log'
                break
        except OSError:
            continue
    if ACTIVE_RUN:
        params_path = ACTIVE_RUN / 'parameters.yaml'
        try:
            params_text = params_path.read_text(errors='replace')
            checkpoints = re.findall(r'^\s*checkpoint:\s*(\d+)\s*$', params_text, re.MULTILINE)
            max_iterations = re.findall(r'^\s*max_iterations:\s*(\d+)\s*$', params_text, re.MULTILINE)
            if checkpoints and max_iterations:
                PHASE_START_ITERATION = int(checkpoints[-1])
                TOTAL_ITERATIONS = PHASE_START_ITERATION + int(max_iterations[-1])
        except OSError:
            pass
        run_name = str(ACTIVE_RUN)
        if 'phase4' in run_name:
            TRAINING_LABEL = 'sim2sim动力学匹配第四阶段训练'
        elif 'phase3b' in run_name:
            TRAINING_LABEL = 'sim2sim鲁棒第三阶段B训练'

PATTERNS = {
    'iteration': r'│\s*iterations\s*│\s*([-+0-9.]+)',
    'reward': r'│\s*train/episode/rew total/mean\s*│\s*([-+0-9.]+)',
    'linear': r'│\s*train/episode/rew tracking lin vel/mean\s*│\s*([-+0-9.]+)',
    'yaw': r'│\s*train/episode/rew tracking ang vel/mean\s*│\s*([-+0-9.]+)',
    'termination': r'│\s*train/episode/number of terminations/mean\s*│\s*([-+0-9.]+)',
    'terrain': r'│\s*train/episode/terrain level/mean\s*│\s*([-+0-9.]+)',
    'stair_success': r'│\s*train/episode/curriculum stair success/mean\s*│\s*([-+0-9.]+)',
    'promotion_rate': r'│\s*train/episode/curriculum promotion rate/mean\s*│\s*([-+0-9.]+)',
    'value_loss': r'│\s*mean value loss/mean\s*│\s*([-+0-9.]+)',
    'kl_max': r'│\s*policy kl max/mean\s*│\s*([-+0-9.]+)',
    'action_saturation': r'│\s*action saturation fraction/mean\s*│\s*([-+0-9.]+)',
    'iter_time': r'│\s*time iter/mean\s*│\s*([-+0-9.]+)',
}


def latest_value(text, key, default=0.0):
    values = re.findall(PATTERNS[key], text)
    return float(values[-1]) if values else default


def process_alive():
    if ACTIVE_PID is not None:
        try:
            os.kill(ACTIVE_PID, 0)
            return True, ACTIVE_PID
        except OSError:
            return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (OSError, ValueError):
        return False, None


def latest_checkpoint():
    checkpoint_root = ACTIVE_RUN if ACTIVE_RUN else ROOT / 'runs'
    items = list(checkpoint_root.glob('**/checkpoints/ac_weights_[0-9]*.pt'))
    return max(items, key=lambda p: p.stat().st_mtime).name if items else '无'


def latest_sim2sim_result():
    if ACTIVE_RUN:
        summaries = list(ACTIVE_RUN.glob('evaluations/sim2sim_stairs/summary.json'))
    else:
        summaries = list((ROOT / 'runs').glob(
            'mybot_v3_sim2sim_phase*/**/evaluations/sim2sim_stairs/summary.json'))
    if not summaries:
        return None
    try:
        rows = json.loads(max(summaries, key=lambda path: path.stat().st_mtime).read_text())
        if not rows:
            return None
        row = rows[-1]
        return {
            'iteration': int(row['iteration']),
            'success_8cm': float(row['success_8cm']),
            'success_10cm': float(row['success_10cm']),
            'success_12cm': float(row['success_12cm']),
            'score': float(row['score']),
        }
    except Exception:
        return None


def gpu_status():
    cmd = [
        'nvidia-smi',
        '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ]
    try:
        fields = [x.strip() for x in subprocess.check_output(cmd, text=True, timeout=15).split(',')]
        return {
            'total': float(fields[0]), 'used': float(fields[1]), 'free': float(fields[2]),
            'util': float(fields[3]), 'temp': float(fields[4]), 'power': float(fields[5]),
        }
    except Exception:
        return None


def build_message():
    text = LOG.read_text(errors='replace') if LOG.exists() else ''
    alive, pid = process_alive()
    iteration = int(latest_value(text, 'iteration'))
    reward = latest_value(text, 'reward')
    linear = latest_value(text, 'linear')
    yaw = latest_value(text, 'yaw')
    termination = latest_value(text, 'termination')
    terrain = latest_value(text, 'terrain')
    stair_success = latest_value(text, 'stair_success')
    promotion_rate = latest_value(text, 'promotion_rate')
    value_loss = latest_value(text, 'value_loss')
    kl_max = latest_value(text, 'kl_max')
    action_saturation = latest_value(text, 'action_saturation')
    iter_time = latest_value(text, 'iter_time')
    percent = iteration / TOTAL_ITERATIONS * 100 if TOTAL_ITERATIONS else 0
    phase_span = max(1, TOTAL_ITERATIONS - PHASE_START_ITERATION)
    phase_percent = max(0.0, min(
        100.0, (iteration - PHASE_START_ITERATION) / phase_span * 100.0))
    step_cm = 5.0 + terrain / 12.0 * 0.9 * 18.0
    eta_h = max(0, TOTAL_ITERATIONS - iteration) * iter_time / 3600 if iter_time > 0 else 0
    errors = []
    checks = [
        ('Traceback', r'Traceback'), ('CUDA OOM', r'CUDA out of memory'),
        ('RuntimeError', r'RuntimeError'), ('NaN', r'(?<![A-Za-z])nan(?![A-Za-z])'),
    ]
    for label, pattern in checks:
        if re.search(pattern, text, re.IGNORECASE):
            errors.append(label)
    if value_loss > 25:
        errors.append(f'value loss异常({value_loss:.1f})')
    gpu = gpu_status()
    sim2sim = latest_sim2sim_result()
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    status = '正常运行' if alive else '训练进程已停止'
    error_text = '未发现NaN、CUDA OOM或运行错误' if not errors else '发现异常：' + '、'.join(errors)
    gpu_text = 'GPU状态读取失败'
    if gpu:
        gpu_text = f"GPU利用率{gpu['util']:.0f}%，显存{gpu['used'] / 1024:.1f}/{gpu['total'] / 1024:.1f}GB，温度{gpu['temp']:.0f}°C"
    sim2sim_text = 'MuJoCo严格验收尚未运行'
    if sim2sim:
        sim2sim_text = (
            f"MuJoCo第{sim2sim['iteration']}轮：8/10/12厘米通过率 "
            f"{sim2sim['success_8cm']:.0f}%/{sim2sim['success_10cm']:.0f}%/"
            f"{sim2sim['success_12cm']:.0f}%（加权{sim2sim['score']:.1f}）")
    message = (
        f'【机器狗{TRAINING_LABEL}进度 {now}】\n'
        f'{status}。当前 {iteration}/{TOTAL_ITERATIONS}（本阶段 {phase_percent:.1f}%，'
        f'总进度 {percent:.1f}%），'
        f'最新检查点 {latest_checkpoint()}。\n'
        f'总奖励 {reward:.3f}，前后速度奖励 {linear:.3f}，转向奖励 {yaw:.3f}，'
        f'近期终止指标 {termination:.3f}。\n'
        f'地形等级 {terrain:.3f}（约{step_cm:.1f}厘米台阶），'
        f'楼梯达标 {100 * stair_success:.1f}%，升级 {100 * promotion_rate:.1f}%。\n'
        f'value loss {value_loss:.3f}，KL {kl_max:.4f}，动作饱和 {100 * action_saturation:.1f}%，'
        f'单次迭代 {iter_time:.3f}秒。\n'
        f'{sim2sim_text}。\n'
        f'{gpu_text}。{error_text}，预计剩余约{eta_h:.1f}小时。'
    )
    status_data = {
        'time': now, 'alive': alive, 'pid': pid, 'iteration': iteration,
        'percent': round(percent, 2), 'phase_percent': round(phase_percent, 2),
        'reward': reward, 'linear': linear,
        'yaw': yaw, 'termination': termination, 'terrain': terrain,
        'stair_success': stair_success, 'promotion_rate': promotion_rate,
        'value_loss': value_loss, 'kl_max': kl_max,
        'action_saturation': action_saturation,
        'step_cm': round(step_cm, 2), 'iter_time': iter_time,
        'checkpoint': latest_checkpoint(), 'gpu': gpu, 'sim2sim': sim2sim, 'errors': errors,
        'eta_hours': round(eta_h, 2), 'message': message,
    }
    return message, status_data


def deliver_pending():
    if not PENDING.exists() or not PENDING.read_text().strip():
        print('No pending training progress message.')
        return 0
    cmd = [
        'timeout', '60', '/home/ldl/.local/bin/hermes', 'send', '--json',
        '--to', TARGET, '--file', str(PENDING),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    combined = (result.stdout + '\n' + result.stderr).strip()
    ok = result.returncode == 0 and '"error"' not in combined and 'Weixin send failed' not in combined
    stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with SEND_LOG.open('a') as fh:
        fh.write(f'{stamp} returncode={result.returncode} ok={ok} {combined[-500:]}\n')
    if ok:
        LAST_SENT.write_text(PENDING.read_text())
        PENDING.unlink(missing_ok=True)
        print('Training progress message delivered through Hermes Weixin.')
        return 0
    print('Hermes Weixin delivery is pending; retry timer will try again.', file=sys.stderr)
    if combined:
        print(combined[-1000:], file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pending-only', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock_path = STATE / 'send.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not args.pending_only:
            message, status = build_message()
            PENDING.write_text(message)
            LAST_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            print(message)
        return deliver_pending()


if __name__ == '__main__':
    raise SystemExit(main())

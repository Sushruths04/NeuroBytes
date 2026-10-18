"""Quickstart for Windows, macOS and Linux (same steps as scripts/reproduce_c13.sh, pure Python, no bash needed).

    python scripts/reproduce_c13.py --data /path/to/eeg-mind-reading-challenge
    python scripts/reproduce_c13.py --data C:\\path\\to\\eeg-mind-reading-challenge --weights C:\\path\\to\\c13o_weights_20261018

Without --weights, the weights archive is downloaded from the GitHub release (with `gh` if installed, otherwise over HTTPS)
and checked against its SHA-256. Prints the development score and writes the submission files to <work>/submission.
"""
import argparse, hashlib, os, shutil, subprocess, sys, tarfile, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH_REPO, TAG, ARCHIVE = 'Sushruths04/neurobytes', 'v1.0', 'c13o_weights_20261018.tar.gz'
SHA256 = 'c81efde41c2ee259160e48c5242d215b1673cc3a6233455ae422937679a3ab3f'


def fetch_weights(work):
    path = os.path.join(work, ARCHIVE)
    if not os.path.exists(path):
        print(f'[1/4] downloading weights (release {TAG}, 161 MB)', flush=True)
        if shutil.which('gh'):
            subprocess.run(['gh', 'release', 'download', TAG, '-R', GH_REPO, '-p', ARCHIVE, '-D', work, '--skip-existing'], check=True)
        else:
            urllib.request.urlretrieve(f'https://github.com/{GH_REPO}/releases/download/{TAG}/{ARCHIVE}', path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    if h.hexdigest() != SHA256:
        sys.exit(f'checksum mismatch for {path}: delete it and retry')
    print(f'{ARCHIVE}: checksum OK', flush=True)
    with tarfile.open(path) as t:
        t.extractall(work)
    return os.path.join(work, 'c13o_weights_20261018')


def run(args, env):
    subprocess.run([sys.executable] + args, cwd=REPO, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='official challenge folder (contains tst/ and tst_stim/)')
    ap.add_argument('--weights', help='extracted c13o_weights_20261018 folder (skips the download)')
    ap.add_argument('--work', default=os.path.join(REPO, 'reproduce'), help='output folder (default: ./reproduce)')
    a = ap.parse_args()
    data, work = os.path.abspath(a.data), os.path.abspath(a.work)
    if not os.path.isdir(os.path.join(data, 'tst')):
        sys.exit(f'{data} has no tst/ folder: pass the official challenge folder with --data')
    os.makedirs(work, exist_ok=True)
    weights = os.path.abspath(a.weights) if a.weights else os.path.join(work, 'c13o_weights_20261018')
    if not os.path.isdir(weights):
        weights = fetch_weights(work)
    env = dict(os.environ, WINFUSE_DATA_ROOT=data, PYTHONPATH=os.pathsep.join([os.path.join(REPO, 'src'), os.environ.get('PYTHONPATH', '')]))
    nets = os.path.join(work, 'nets')
    print('[2/4] scoring raw test EEG with the two networks (h6, h10; 5 seeds each)', flush=True)
    for tag in ('h6', 'h10'):
        for s in range(1, 7):
            run(['combo/nc_predict.py', '--weights', weights, '--tag', tag, '--subject', f'sub-0{s}',
                 '--eeg', os.path.join(data, 'tst', f'sub-0{s}_tst.npy'), '--out', nets], env)
    env.update(NC_DIR=nets, NC_FAMILIES='h10:0.075,h6:0.075', C13_ALIGN_TO_DATA='1')
    if os.path.isdir(os.path.join(data, 'tst_stim')):
        print('[3/4] development test; expected: top1 36.03 top5 67.16 rank 5.99 (GPU) / 6.00 (CPU)', flush=True)
        run(['combo/c11.py', '--dev'], env)
    else:
        print('[3/4] no tst_stim/ folder: development score skipped', flush=True)
    out = os.path.join(work, 'submission')
    print('[4/4] submission files', flush=True)
    run(['combo/c11.py', '--out', out], env)
    run(['combo/c11.py', '--check', out], env)


if __name__ == '__main__':
    main()

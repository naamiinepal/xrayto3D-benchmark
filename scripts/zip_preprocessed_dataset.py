from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('src')
parser.add_argument('dest')
parser.add_argument('--dry-run','-n',action='store_true',default=False)
parser.add_argument('--exclude-from','-e')

args = parser.parse_args()


print(args)

def get_rsync_cp_dir_cmd(src_dir, dest_dir, dry_run=False, verbose=True,exclude_from=None):
    base_cmd = f'rsync -a {src_dir} {dest_dir}'
    if verbose:
        base_cmd += ' -v'
    if dry_run:
        base_cmd += ' --dry-run'
    if exclude_from:
        base_cmd += f' --exclude-from={exclude_from}'
    return base_cmd

def run_os_cmd(cmd_str):
    os.system(cmd_str)

rsync_cmd = get_rsync_cp_dir_cmd(args.src, args.dest,dry_run=args.dry_run,exclude_from=args.exclude_from)
print(rsync_cmd)
run_os_cmd(rsync_cmd)
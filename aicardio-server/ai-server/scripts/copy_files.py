import shutil
import os
import glob

def copy_file(path, dst_dir):
  print(path, dst_dir)
  dirname = os.path.dirname(path)
  pos = dirname.find("EF")
  if pos == -1: return
  reldir = dirname[pos:]
  print(reldir)
  dst_reldir = os.path.join(dst_dir, reldir)
  dst_path = os.path.join(dst_reldir, os.path.basename(path))
  print(dst_reldir, dst_path)
  os.makedirs(dst_reldir, exist_ok=True)
  shutil.copy(path, dst_path)

def copy_all(src_dir, dst_dir, exts = ['txt', 'gif']):
  for path in sum([glob.glob(os.path.join(src_dir, '**', f'*.{e}'), recursive=True) for e in exts], []):
    copy_file(path, dst_dir)

if __name__ == "__main__":
  #copy_file("visualize/data.local/data/EF/0065__STE_AN/IM_0002_2C/ef_gls.txt",
  #          "summary")
  copy_all("visualize-20200315", "summary-20200315")

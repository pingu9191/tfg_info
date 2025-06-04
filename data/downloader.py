import os
import gzip
import shutil

def downloader(line_number: int):
    # Open index.txt
    with open('data/index.txt', 'r') as f:
        # Read the file line by line
        lines = f.readlines()
        
        if line_number < 0 or line_number >= len(lines):
            print("Invalid line number.")
            return False
        
        line = lines[line_number].strip()
        print(line)
        kk = line.split('/')[-1].replace('\n', '')
        print(kk)
        os.system(f'curl -o data/{kk} {line}')
        
        # Unzip .gz
        if kk.endswith('.gz'):
            with gzip.open(f'data/{kk}', 'rb') as f_in:
                with open(f'data/{kk[:-3]}', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Unzipped {kk} to {kk[:-3]}')
            os.remove(f'data/{kk}')
        else:
            print(f'Skipped unzipping for {kk}, not a .gz file')
            
        return kk[:-3]
                
if __name__ == "__main__":
    ls = downloader(1)
    print("Download complete. File saved as:", ls)
            
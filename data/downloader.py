import os
import gzip
import shutil

# Open index.txt
with open('data/index.txt', 'r') as f:
    # Read the file line by line
    count = 0
    for line in f:
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
            count += 1
            os.remove(f'data/{kk}')
        else:
            print(f'Skipped unzipping for {kk}, not a .gz file')
        
        if count == 1:
            exit(0)
        # Strip whitespace and check if the line is not empty
        """count = 0
        # Download the file using wget
        os.system(f'wget -c {line} -O {count+1}.csv')
        print(f'Downloaded {count+1}.csv from {line.strip()}')
        exit(0)"""
            


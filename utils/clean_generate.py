import os
import sys
def check_and_remove_empty_fasta(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.fasta', '.fa')):
                file_path = os.path.join(root, file)
                
                # check
                if os.path.getsize(file_path) == 0:
                    print(f"empty: {file_path}")
                    try:
                        os.remove(file_path)
                        print(f"delete: {file_path}")
                    except Exception as e:
                        print(f"error ({file_path}): {str(e)}")

if __name__ == "__main__":
    target_dir = sys.argv[1]
    check_and_remove_empty_fasta(target_dir)
    print("Done!")
import os
import re

def process_file(md_path, ipynb_name, module_name):
    # Construct github relative path
    repo_owner = "vinod-seth"
    repo_name = "slm-development"
    colab_url = f"https://colab.research.google.com/github/{repo_owner}/{repo_name}/blob/main/tutorial/{module_name}/{ipynb_name}"
    badge_markdown = f"\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n"
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "colab-badge.svg" in content:
        print(f" -> Badge already present in {os.path.basename(md_path)}")
        return False
        
    lines = content.splitlines(keepends=True)
    if not lines:
        return False
        
    # Insert badge below the title (the first line)
    # Ensure the first line is actually a title starting with #
    if lines[0].startswith("#"):
        # Insert the badge with empty lines surrounding it for markdown compatibility
        lines.insert(1, badge_markdown)
        updated_content = "".join(lines)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f" -> Added badge to {os.path.basename(md_path)}")
        return True
    else:
        print(f" -> Warning: First line of {os.path.basename(md_path)} is not a title. Skipping.")
        return False

def main():
    tutorial_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tutorial")
    if not os.path.exists(tutorial_dir):
        tutorial_dir = "./tutorial"
        
    print(f"Scanning tutorial directory: {tutorial_dir}")
    count = 0
    
    for root, dirs, files in os.walk(tutorial_dir):
        module_name = os.path.basename(root)
        md_files = [f for f in files if f.endswith(".md") and not f.startswith("README") and not f.startswith("quiz")]
        ipynb_files = [f for f in files if f.endswith(".ipynb")]
        
        for md_file in md_files:
            # Match by prefix (e.g. "01_")
            match = re.match(r"^(\d+)_", md_file)
            if match:
                prefix = match.group(1)
                matching_ipynb = [ip for ip in ipynb_files if ip.startswith(prefix)]
                if matching_ipynb:
                    ipynb_name = matching_ipynb[0]
                    md_path = os.path.join(root, md_file)
                    if process_file(md_path, ipynb_name, module_name):
                        count += 1
                else:
                    print(f" -> No matching notebook found for {md_file} with prefix {prefix}")
            else:
                print(f" -> Skip unmatched file name format: {md_file}")
                
    print(f"Finished. Added badges to {count} files.")

if __name__ == "__main__":
    main()

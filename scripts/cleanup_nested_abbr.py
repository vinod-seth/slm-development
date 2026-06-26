import os
import re

# Regex to match nested abbr tags and capture the outermost title and the innermost text
nested_abbr_regex = re.compile(
    r'<abbr\b[^>]*title="([^"]+)"[^>]*>(?:<abbr\b[^>]*>)+([^<]+)(?:</abbr>)+',
    re.IGNORECASE
)

def cleanup_content(content):
    prev = ""
    while prev != content:
        prev = content
        content = nested_abbr_regex.sub(r'<abbr title="\1">\2</abbr>', content)
    return content

def main():
    tutorial_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tutorial")
    if not os.path.exists(tutorial_dir):
        tutorial_dir = "./tutorial"
        
    print(f"Scanning directory for nested tags: {tutorial_dir}")
    count = 0
    for root, dirs, files in os.walk(tutorial_dir):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                cleaned = cleanup_content(content)
                if content != cleaned:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    print(f" -> Cleaned nested tags in {os.path.basename(filepath)}")
                    count += 1
                    
    print(f"Finished cleanup. Cleaned {count} files.")

if __name__ == "__main__":
    main()

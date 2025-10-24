from pathlib import Path

def print_tree(path: Path, prefix: str = ""):
    """Recursively print directory tree"""
    if not path.exists():
        print(f"[!] Path not found: {path}")
        return
    if path.is_file():
        print(prefix + path.name)
        return

    print(prefix + f"{path.name}/")
    for child in sorted(path.iterdir()):
        if child.name.startswith(".") or child.name.startswith("__pycache__"):
            continue
        if child.is_dir():
            print_tree(child, prefix + "    ")
        else:
            print(prefix + "    " + child.name)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(f"Project root: {root}\n")
    print_tree(root)

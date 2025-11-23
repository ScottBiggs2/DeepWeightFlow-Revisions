import os
from pathlib import Path

def renumber_model_files(directory='dummy_imagenet_resnet_models', 
                         old_start=251, 
                         old_end=500, 
                         new_start=0):
    """
    Renumber model files from resnet_weights_{old_start-old_end}.pt 
    to resnet_weights_{new_start-...}.pt
    
    Args:
        directory: Directory containing the .pt files
        old_start: Starting index of existing files (inclusive)
        old_end: Ending index of existing files (inclusive)
        new_start: Starting index for renumbered files
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        return
    
    # Collect all files that need renaming
    files_to_rename = []
    for i in range(old_start, old_end + 1):
        old_name = f'resnet_weights_{i}.pt'
        old_path = directory / old_name
        if old_path.exists():
            files_to_rename.append((i, old_path))
    
    if not files_to_rename:
        print(f"No files found in range {old_start}-{old_end}")
        return
    
    print(f"Found {len(files_to_rename)} files to rename")
    print(f"Renumbering from {old_start}-{old_end} → {new_start}-{new_start + len(files_to_rename) - 1}")
    
    # First pass: rename to temporary names to avoid conflicts
    temp_renames = []
    for old_idx, old_path in files_to_rename:
        temp_name = f'resnet_weights_TEMP_{old_idx}.pt'
        temp_path = directory / temp_name
        old_path.rename(temp_path)
        temp_renames.append((old_idx, temp_path))
    
    print("✓ Renamed to temporary files")
    
    # Second pass: rename to final names
    new_idx = new_start
    for old_idx, temp_path in temp_renames:
        new_name = f'resnet_weights_{new_idx}.pt'
        new_path = directory / new_name
        temp_path.rename(new_path)
        print(f"  {old_idx} → {new_idx}")
        new_idx += 1
    
    print(f"\n✓ Successfully renumbered {len(files_to_rename)} files!")
    print(f"Files now range from resnet_weights_{new_start}.pt to resnet_weights_{new_start + len(files_to_rename) - 1}.pt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Renumber ResNet model checkpoint files')
    parser.add_argument('--directory', type=str, default='dummy_imagenet_resnet_models',
                       help='Directory containing the .pt files')
    parser.add_argument('--old_start', type=int, default=251,
                       help='Starting index of existing files (inclusive)')
    parser.add_argument('--old_end', type=int, default=500,
                       help='Ending index of existing files (inclusive)')
    parser.add_argument('--new_start', type=int, default=0,
                       help='Starting index for renumbered files')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview changes without actually renaming')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - No files will be modified")
        directory = Path(args.directory)
        count = 0
        for i in range(args.old_start, args.old_end + 1):
            old_name = f'resnet_weights_{i}.pt'
            if (directory / old_name).exists():
                new_idx = args.new_start + count
                print(f"  Would rename: {old_name} → resnet_weights_{new_idx}.pt")
                count += 1
        print(f"\nTotal files that would be renamed: {count}")
    else:
        renumber_model_files(
            directory=args.directory,
            old_start=args.old_start,
            old_end=args.old_end,
            new_start=args.new_start
        )
"""
Resume training from a saved checkpoint
This is a convenience script to quickly resume training
"""

import os
import glob
from train_2048 import main


def find_latest_checkpoint(models_dir='models'):
    """Find the latest checkpoint file."""
    checkpoints = glob.glob(os.path.join(models_dir, 'dqn_2048_ep*.pth'))
    
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}/")
        return None
    
    # Sort by episode number
    checkpoints.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
    
    return checkpoints[-1]


def resume_training(checkpoint_path=None):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to specific checkpoint, or None to use latest
    """
    
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("No checkpoint found. Starting fresh training...")
            main(resume_from=None)
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoints = glob.glob('models/dqn_2048_ep*.pth')
        for cp in sorted(checkpoints):
            print(f"  - {cp}")
        return
    
    print(f"\nResuming training from: {checkpoint_path}")
    main(resume_from=checkpoint_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume 2048 DQN Training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint (default: use latest)')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable checkpoints:")
        checkpoints = glob.glob('models/dqn_2048_ep*.pth')
        if checkpoints:
            for cp in sorted(checkpoints):
                size_mb = os.path.getsize(cp) / (1024 * 1024)
                print(f"  - {cp} ({size_mb:.2f} MB)")
        else:
            print("  No checkpoints found in models/")
    else:
        resume_training(args.checkpoint)

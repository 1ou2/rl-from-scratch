# Creating a Video Showing Model Progress

## Quick Start

### 1. Demo with automatic checkpoint selection (6 evenly-spaced checkpoints)
```bash
python demo_progress.py
```

### 2. Demo ALL checkpoints
```bash
python demo_progress.py --all
```

### 3. Demo specific checkpoints
```bash
python demo_progress.py --checkpoints models/dqn_2048_ep500.pth models/dqn_2048_ep2500.pth models/dqn_2048_ep5000.pth
```

### 4. Record frames for video creation
```bash
python demo_progress.py --all --record
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoints` | Specific checkpoint files | Auto-select 6 |
| `--all` | Use all available checkpoints | False |
| `--interval N` | Use every Nth checkpoint | 1 |
| `--games N` | Games to play per checkpoint | 3 |
| `--delay MS` | Delay between moves (ms) | 300 |
| `--record` | Save frames for video | False |
| `--output-dir` | Frame output directory | demo_frames |

## Examples

### Show progress from early to final training
```bash
# Select key milestones
python demo_progress.py --checkpoints \
    models/dqn_2048_ep500.pth \
    models/dqn_2048_ep1000.pth \
    models/dqn_2048_ep2000.pth \
    models/dqn_2048_ep3000.pth \
    models/dqn_2048_ep4000.pth \
    models/dqn_2048_ep5000.pth
```

### Record video with every 500 episodes
```bash
# If you have ep500, ep1000, ep1500, etc.
python demo_progress.py --all --interval 1 --record --delay 200
```

### Quick comparison (fast playback)
```bash
python demo_progress.py --all --games 1 --delay 100
```

### Slow, detailed view (good for recording)
```bash
python demo_progress.py --all --games 2 --delay 500 --record
```

## Creating the Video

After recording with `--record`, create a video using ffmpeg:

```bash
# Standard quality (recommended)
ffmpeg -framerate 30 -i demo_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p \
       -crf 18 \
       model_progress.mp4

# High quality
ffmpeg -framerate 30 -i demo_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p \
       -crf 15 \
       model_progress_hq.mp4

# Fast encode (lower quality)
ffmpeg -framerate 30 -i demo_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p \
       -preset fast -crf 23 \
       model_progress_fast.mp4

# With speedup (2x speed)
ffmpeg -framerate 60 -i demo_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p \
       -crf 18 \
       model_progress_2x.mp4
```

## What You'll See

The demo will:
1. Load each checkpoint in order
2. Play N games with each model
3. Display:
   - Current checkpoint (episode number)
   - Game number
   - Current max tile
   - Score
   - Game board with all tiles

Each checkpoint will show the model's performance at that training stage.

## Tips

### For a Great Video:

1. **Select meaningful checkpoints:**
   - Start with early training (ep500 or ep1000)
   - Include middle stages (ep2000, ep3000)
   - End with final model (ep5000)

2. **Adjust speed for clarity:**
   - Use `--delay 500` for slower, clearer gameplay
   - Use `--delay 100` for faster videos
   - Edit video speed in post if needed

3. **Multiple games per checkpoint:**
   - `--games 3` shows consistency
   - `--games 1` makes shorter videos

4. **Frame rate for video:**
   - 30 fps is standard
   - 60 fps for smoother playback
   - Adjust framerate in ffmpeg command

### Example Workflow:

```bash
# 1. Record frames
python demo_progress.py \
    --checkpoints models/dqn_2048_ep500.pth \
                  models/dqn_2048_ep1500.pth \
                  models/dqn_2048_ep2500.pth \
                  models/dqn_2048_ep3500.pth \
                  models/dqn_2048_ep5000.pth \
    --games 2 \
    --delay 250 \
    --record

# 2. Create video
ffmpeg -framerate 30 -i demo_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p \
       -crf 18 \
       2048_training_progress.mp4

# 3. Clean up frames (optional)
rm -rf demo_frames/
```

## Keyboard Controls During Demo

- **ESC**: Stop demo and exit
- **Close Window**: Stop demo

## Output

The demo will print:
- Progress for each checkpoint
- Statistics per game (score, max tile, moves)
- Summary of each checkpoint's performance
- Overall progress comparison
- Video creation instructions (if recording)

## Troubleshooting

**No checkpoints found:**
```bash
# Check what checkpoints you have
ls -lh models/dqn_2048_ep*.pth

# Make sure you've trained a model first
python train_2048.py
```

**Video creation fails:**
```bash
# Install ffmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**Demo too slow:**
```bash
# Reduce delay between moves
python demo_progress.py --delay 50
```

**Demo too fast:**
```bash
# Increase delay between moves
python demo_progress.py --delay 800
```

## Advanced: Custom Video Editing

After creating the base video, you can enhance it with:

1. **Add title slides:**
   - Use video editor to add intro/outro
   - Show episode numbers as overlays

2. **Side-by-side comparison:**
   - Record early and late checkpoints separately
   - Combine videos side-by-side

3. **Add audio/music:**
   ```bash
   ffmpeg -i model_progress.mp4 -i music.mp3 \
          -c:v copy -c:a aac -shortest \
          model_progress_with_music.mp4
   ```

---

**Ready to create your demo video?**
```bash
python demo_progress.py --all --record
```

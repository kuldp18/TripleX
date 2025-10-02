# Additional Instructions Feature Usage Guide

This guide explains how to use the additional instructions feature in `joycaption2_video.py`.

## Overview

The script now supports adding custom instructions to both:

1. **Frame captioning** - Instructions applied to each individual frame
2. **Composite captioning** - Instructions applied when generating the final video description

## API Design

### Command-Line Arguments

- `--frame_instructions`: Additional instructions for frame captioning
- `--composite_instructions`: Additional instructions for composite caption generation

Both arguments accept:

- **Inline text** (including multi-line text)
- **File paths** to text files containing instructions

## Usage Examples

### 1. Inline Instructions (Single Line)

```bash
python captioners/joycaption2_video.py --dir ./videos --frame_instructions "Focus on facial expressions and emotions"
```

### 2. Inline Instructions (Multi-Line)

On Windows PowerShell, use backticks or quotes:

```powershell
python captioners/joycaption2_video.py --dir ./videos `
  --frame_instructions "Focus on the following aspects:
- Facial expressions
- Body language
- Color palette" `
  --composite_instructions "Create a narrative-style description that emphasizes emotional tone"
```

Or using string literals:

```powershell
python captioners/joycaption2_video.py --dir ./videos --frame_instructions "Focus on facial expressions`nMention any props or objects`nDescribe the mood"
```

### 3. File-Based Instructions

Create instruction files:

**frame_instructions.txt:**

```
Focus on the following aspects when describing each frame:
- Identify any people and their activities
- Describe the setting and environment
- Note any significant objects or props
- Comment on the lighting and color grading
- Describe the camera angle and composition
- Mention any text visible in the frame (despite the default instruction to ignore it)
```

**composite_instructions.txt:**

```
When creating the final video description:
- Write in a cinematic style
- Emphasize the narrative arc
- Connect the scenes logically
- Mention any recurring themes or motifs
- Keep the tone engaging and descriptive
- Aim for 3-4 sentences maximum
```

Then run:

```bash
python captioners/joycaption2_video.py --dir ./videos --frame_instructions frame_instructions.txt --composite_instructions composite_instructions.txt
```

### 4. Mixed Usage

You can use inline for one and file for another:

```bash
python captioners/joycaption2_video.py --dir ./videos --frame_instructions "Be very detailed and specific" --composite_instructions composite_instructions.txt
```

### 5. Same Instructions for Both

If you want the same additional instructions for both frame and composite:

```bash
python captioners/joycaption2_video.py --dir ./videos --frame_instructions instructions.txt --composite_instructions instructions.txt
```

Or inline:

```bash
python captioners/joycaption2_video.py --dir ./videos --frame_instructions "Use formal language" --composite_instructions "Use formal language"
```

## How It Works

1. The `load_instructions()` function checks if the argument is a file path
2. If it's a valid file, it reads the content from the file
3. If it's not a file, it treats the argument as inline text
4. The instructions are appended to the existing prompts with a clear "Additional instructions:" header
5. Empty or None values are handled gracefully (no instructions added)

## Implementation Details

### Frame Captioning

Original prompt:

```
Write a detailed descriptive caption for this video frame at {timestamp} seconds.
Include information about the scene, people, actions, lighting, and camera angle.
Use plain, everyday language. Do NOT mention any text that is in the image.
```

With additional instructions:

```
Write a detailed descriptive caption for this video frame at {timestamp} seconds.
Include information about the scene, people, actions, lighting, and camera angle.
Use plain, everyday language. Do NOT mention any text that is in the image.

Additional instructions:
{your custom instructions here}
```

### Composite Captioning

Similar pattern - your instructions are appended after the base prompt.

## Benefits

- **Flexibility**: Choose between inline or file-based instructions
- **Reusability**: Save frequently-used instructions in files
- **Independence**: Frame and composite instructions can be different or the same
- **Backward Compatible**: Scripts without these arguments work exactly as before
- **Multi-line Support**: Handle complex, detailed instructions spanning multiple lines

## Tips

1. **Use files for complex instructions** - Easier to edit and reuse
2. **Use inline for simple tweaks** - Quick adjustments without file management
3. **Test iteratively** - Start with simple instructions and refine
4. **Be specific** - More specific instructions generally produce better results
5. **Check logs** - The first 100 characters of loaded instructions are logged for verification

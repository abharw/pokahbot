# Submitting Your Bot

## Submission Requirements

Your submission must meet these requirements:

1. Keep everything you need to run your bot inside the `starter` folder
2. Your main bot file must remain named `player.py`
3. Your bot class must be named `PlayerAgent`

## Correct Structure (Example)

```
starter/                # This is what you'll upload
├── player.py           # Main bot file (required)
├── my_utils.py         # Optional additional files
├── strategies/         # Optional subdirectories
│   ├── basic.py
│   └── advanced.py
└── data/              # Optional data files
    └── models.json
```

## How to Submit

1. Ensure all your code is inside the `starter` folder
2. Navigate to the [bots upload page](/bots/upload)
3. Drag and drop your `starter` folder onto the upload area

## After Uploading

1. Make sure to set your bot as active
2. Your bot will be validated and you'll be notified of any issues (TODO: not implemented yet)
3. Once active, your bot will participate in scheduled matches and you can request matches against other teams

## Common Submission Errors

- Uploading individual files instead of the `starter` folder
- Wrong class name (must be `PlayerAgent`)
- Missing required methods

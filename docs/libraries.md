# Allowed Python Libraries

## Environment Specifications

- Python 3.12 is required
- All bots run in isolated Python 3.12 virtual environments
- Libraries are pre-installed in the tournament environment

## Pre-installed Libraries

The following libraries are pre-installed in the tournament environment and are safe to use in your bot:

### Core Scientific Stack

- **numpy** (1.26.0)
- **pandas** (2.2.0)
- **scipy** (1.14.0)
- **scikit-learn** (1.5.0)

### Machine Learning

- **torch** (2.2.0)
- **tensorflow** (2.16.0)
- **keras** (3.7.0)
- **stable-baselines3** (2.4.0)

### Game Engine Dependencies

- **gymnasium** (0.26.0)
- **treys** (0.1.0)

### Utility Libraries

- **tqdm** (4.67.0)
- **joblib** (1.4.0)
- **pickle** (built-in)
- All Python standard library modules

### API Dependencies

- **fastapi** (0.114.0)
- **pydantic** (2.9.0)
- **requests** (2.32.0)
- **uvicorn** (0.30.0)

## Restrictions

### Not Allowed

- External network calls
- File system access outside the bot's directory
- System calls or subprocess execution
- Libraries not listed above

### Memory Usage

- Your bot must stay within the 4GB RAM limit
- Large model files count towards this limit
- Consider using smaller, optimized models

### Storage

- Your submission can include additional files (e.g., trained models)
- Total submission size must be under 10MB (to be changed to 500MB)
- Files must be read during bot initialization, not during gameplay

## Best Practices

1. **Initialization**
   - Load models and heavy resources during bot initialization
   - Cache calculations that will be reused across hands

2. **Memory Management**
   - Monitor your bot's memory usage during testing
   - Clear unnecessary variables between hands
   - Use efficient data structures

3. **Version Compatibility**
   - Test your bot with the exact library versions listed
   - Avoid using bleeding-edge features that might not be available

## Adding New Libraries

If you believe a library should be added to the allowed list:

1. Open an issue on the tournament GitHub repository
2. Provide justification for the library's inclusion
3. Ensure it doesn't provide unfair advantages
4. Consider memory and computational impact

> **Note**: Library versions will remain fixed throughout the tournament to ensure consistency.

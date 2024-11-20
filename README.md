# AnimateDiff on Intel GPUs
> Turn your prompts into magical animations with AnimateDiff! Now optimized for Intel GPUs

## Quick Deploy
1. Get an Intel Data Centre GPU Max VM (48GB recommended for production)
   ```bash
   # Head over to https://cloud.intel.com
   # Sign up for a standard account
   # Spin up a Max Series GPU VM
   ```

2. SSH Connection Setup
   ```bash 
   # Option 1: Direct tunnel
   ssh -L 8000:localhost:8000 guest@ip1 username@ip2

   # Option 2: Cloudflare tunnel (recommended for prod)
   # Install cloudflared .deb package from https://pkg.cloudflare.com/
   # Start tunnel:
   cloudflared tunnel --url http://localhost:8000
   ```

3. Run the server
   ```bash
   python3.12 server.py
   ```

4. Generate animations
   ```bash
   python3.12 client.py "((best quality)), ((masterpiece)), ((realistic)), flowing neon blue hair, cybergirl, chrome battle suit, determined expression, high-resolution, cyberpunk street"
   ```

## System Requirements
- Intel Data Centre GPU Max Series (48GB recommended for concurrent loads)
- Smaller Intel GPUs work fine for testing/personal use
- Python 3.12+

## Sample Prompts
```python
# Cyberpunk vibes
"((best quality)), ((masterpiece)), ((realistic)), silver-white braided hair, cybergirl, iridescent power armor, relaxed pose, high-resolution, tech laboratory, glowing circuits, gentle smile"

# Space fantasy
"((best quality)), ((masterpiece)), ((realistic)), purple gradient hair, cybergirl, sleek nanofiber suit, elegant posture, high-resolution, space station interior, floating holograms, mysterious expression"
```

## Quick Tips
- Use the viewer.py to see your generated animations
- Higher VRAM = More concurrent generations
- Keep prompts detailed but focused

## Need Help?
Check out Intel's Tiber AI Cloud docs (https://www.intel.com/content/www/us/en/developer/tools/tiber/ai-cloud.html) for more details on VM setup and management.

## License
Apache License 2.0

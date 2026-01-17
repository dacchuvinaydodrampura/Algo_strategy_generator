"""
Git Sync Module - Powers the Permanent Memory.

Provides functionality to push brain memory and strategy library
back to GitHub automatically after every research cycle.
"""

import os
import subprocess
import logging
from config import Config

logger = logging.getLogger(__name__)

def sync_to_github():
    """
    Commit and push strategies.json and dna_memory.json to GitHub.
    Uses GITHUB_TOKEN if available for authentication on Render.
    """
    if not Config.SYNC_ENABLED:
        return False

    if not os.path.exists(Config.STORAGE_PATH) and not os.path.exists(Config.DNA_MEMORY_PATH):
        logger.warning("📝 GitSync: No data files found to sync.")
        return False

    try:
        # 1. Configure Git (Required for Render environments)
        # These are local config calls, safe for 512MB RAM
        subprocess.run(["git", "config", "--global", "user.email", "bot@tradersdna.ai"], check=False)
        subprocess.run(["git", "config", "--global", "user.name", "ExperienceBot"], check=False)

        # 2. Update Remote URL with Token (if available)
        if Config.GITHUB_TOKEN:
            try:
                # Try to get existing URL
                remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                # Fallback: Try to find any remote URL
                try:
                    remotes = subprocess.check_output(["git", "remote", "-v"]).decode()
                    import re
                    match = re.search(r"https://github\.com/[\w\-/.]+", remotes)
                    if match:
                        remote_url = match.group(0)
                    else:
                        raise ValueError("No github remote found")
                except Exception:
                    # Final Fallback: Use the known repository URL
                    remote_url = "https://github.com/dacchuvinadodrampura/Algo_strategy_generator.git"

            # Inject the token for authenticated push
            if "https://github.com/" in remote_url:
                new_url = remote_url.replace("https://github.com/", f"https://{Config.GITHUB_TOKEN}@github.com/")
                # Ensure 'origin' exists or set it correctly
                subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", new_url], check=True, capture_output=True)
            elif Config.GITHUB_TOKEN in remote_url:
                # Token already there, just ensure origin is correct
                subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", remote_url], check=True, capture_output=True)

        # 3. Add Files
        files_to_sync = []
        if os.path.exists(Config.STORAGE_PATH): files_to_sync.append(Config.STORAGE_PATH)
        if os.path.exists(Config.DNA_MEMORY_PATH): files_to_sync.append(Config.DNA_MEMORY_PATH)
        
        if not files_to_sync:
            return False

        subprocess.run(["git", "add", "-f"] + files_to_sync, check=True, capture_output=True)

        # 4. Commit
        from strategy_repository import get_stored_count
        count = get_stored_count()
        commit_msg = f"🤖 Auto-Sync: Intelligence Memory Update [Lib: {count}]"
        
        # Check if there are actual changes before committing
        status = subprocess.run(["git", "status", "--short"], capture_output=True).stdout.decode()
        if not status:
            logger.info("🔄 GitSync: No memory changes detected.")
            return True

        subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)

        # 5. Push
        # We use --force if on Render to ensure memory is always canonical
        push_cmd = ["git", "push"]
        result = subprocess.run(push_cmd, check=True, capture_output=True)
        
        logger.info(f"💾 GitSync: Experience pushed to GitHub. [Lib Size: {count}]")
        return True

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"❌ GitSync Failed: {error_msg}")
        return False
    except Exception as e:
        logger.error(f"❌ GitSync Error: {e}")
        return False

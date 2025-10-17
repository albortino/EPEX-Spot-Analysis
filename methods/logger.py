import os
import inspect
from datetime import datetime
from methods.config import DEBUG

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir="logs", date_format="%Y-%m-%d %H:%M:%S", debug: bool = False):
        # The initialized check prevents re-running __init__ on the singleton instance
        if not hasattr(self, 'initialized'):
            self.log_dir = log_dir
            self.date_format = date_format
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.initialized = True
            self.debug = debug

    @property
    def _get_log_filepath(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"log_{today}.txt")

    def log(self, message: str, severity: int = 0):
        """
        Logs a message to the console and optionally to a file.
        Severity 0: Console only (for UI rendering, etc.).
        Severity 1: Console and file (for important events like API calls, file parsing).
        """
        now = datetime.now().strftime(self.date_format)
        
        # Get the filename of the caller for context
        try:
            caller_filename = os.path.basename(inspect.stack()[1].filename)
            log_prefix = f"{now} [{caller_filename}]"
        except (IndexError, AttributeError):
            log_prefix = f"{now}"
        
        log_message = f"{log_prefix}: {message}"
        print(log_message)

        if severity >= 1:
            try:
                with open(self._get_log_filepath, "a", encoding="utf-8") as f:
                    f.write(log_message + "\n")
            except Exception as e:
                print(f"{now} [logger.py]: CRITICAL - Failed to write to log file. Error: {e}")

# Singleton instance to be imported and used across the application
logger = Logger(debug=DEBUG)


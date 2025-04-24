import asyncio
import os
import shlex
from typing import Optional, Dict
from app.tool.base import BaseTool, CLIResult
from app.tool.color import Color
from app.prompt.lerobot import ACTIONBASE as ActionBase
from app.config import config
import subprocess  # 已在顶部导入

CLI = '''
huggingface-cli login --token {token} --add-to-git-credential && \
export HF_USER=$(huggingface-cli whoami | head -n 1) && \
cd ~/lerobot && \
/home/chudy/anaconda3/envs/lerobot/bin/python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id={HF_USER}/{Action} \
  --control.episode=0
'''

# CLI= '''
# python -u robot.py \
#  --action="{Action}"
# '''


def get_hf_user() -> str:
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout.strip().splitlines()[0]
    except subprocess.CalledProcessError as e:
        print("Error getting HF username:", e.stderr)
        return ""



actions:Dict[int,str] = config.action_src.actions


class RobotAction(BaseTool):
    name: str = "Robot_action"
    description: str = f""" Robot action execution tool, used to control the robot to complete 20 predefined daily action tasks.
Users need to provide an action ID between 1 and {config.action.count}, and the tool will automatically execute the corresponding robot control script.
{config.action.format_for_prompt()}
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "action_id": {
                "type": "int",
                "minimum": 1,
                "maximum": config.action.count,
                "description": "Predefined action ID numbers (integers between 1 and 20)",
            }
        },
        "required": ["action_id"],
    }
    process: Optional[asyncio.subprocess.Process] = None
    current_path: str = os.getcwd()
    lock: asyncio.Lock = asyncio.Lock()

    async def execute(self, action_id: int) -> CLIResult:
        if action_id is None:
            return CLIResult(output="", error="Missing action ID parameter")

        if action_id not in actions:
            return CLIResult(output="", error=f"Invalid action ID: {action_id} (valid range: 1-20)")

        action_desc = actions[action_id]
        safe_action = shlex.quote(action_desc)

        hf_user = get_hf_user()
        # #添加 subprocess 调用本地脚本（比如 run_control.sh）
        # subprocess.run(["bash", "run_control.sh", action_desc])

        command = CLI.format(Action=safe_action, HF_USER=hf_user)
        print(Color.CYAN,"Command:\n================================= \n",command,Color.RESET)
        # final_output = await self.execute_in_env("open_manus",command)
        final_output = await self._execute(command)
        return final_output

    async def _execute(self, command: str) -> CLIResult:
        """
        Execute a terminal command asynchronously with persistent context.

        Args:
            command (str): The terminal command to execute.

        Returns:
            str: The output, and error of the command execution.
        """
        # Split the command by & to handle multiple commands
        commands = [cmd.strip() for cmd in command.split("&") if cmd.strip()]
        final_output = CLIResult(output="", error="")
        for cmd in commands:
            sanitized_command = self._sanitize_command(cmd)
            # Handle 'cd' command internally
            if sanitized_command.lstrip().startswith("cd "):
                result = await self._handle_cd_command(sanitized_command)
            else:
                async with self.lock:
                    try:
                        self.process = await asyncio.create_subprocess_shell(
                            sanitized_command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=self.current_path,
                            ## windows need
                            shell=True,
                        )
                        stdout, stderr = [], []
                        async for line in self.process.stdout:
                            decoded = line.decode(errors="replace").strip()
                            stdout.append(decoded)
                            print(Color.YELLOW,"stdout :",Color.RESET,decoded)
                        async for line in self.process.stderr:
                            decoded = line.decode(errors="replace").strip()
                            stderr.append(decoded)
                            print(Color.YELLOW,"stderr :",Color.RESET,decoded)

                        await self.process.wait()
                        # stdout, stderr = await self.process.communicate()
                        result = CLIResult(
                            output="\n".join(stdout),
                            error="\n".join(stderr)
                        )
                        print(final_output)
                    except Exception as e:
                        result = CLIResult(output="", error=str(e))
                        print("error : ",e)
                    finally:
                        self.process = None

            # Combine outputs
            if result.output:
                final_output.output += (
                    (result.output + "\n") if final_output.output else result.output
                )
                print()
            if result.error:
                final_output.error += (
                    (result.error + "\n") if final_output.error else result.error
                )

        # Remove trailing newlines
        final_output.output = final_output.output.rstrip()
        final_output.error = final_output.error.rstrip()
        return final_output

    async def execute_in_env(self, env_name: str, command: str) -> CLIResult:
        """
        Execute a terminal command asynchronously within a specified Conda environment.

        Args:
            env_name (str): The name of the Conda environment.
            command (str): The terminal command to execute within the environment.

        Returns:
            str: The output, and error of the command execution.
        """
        sanitized_command = self._sanitize_command(command)

        # Construct the command to run within the Conda environment
        # Using 'conda runhe current active token is: `lerobot aloha` -n env_name command' to execute without activating
        conda_command = f"conda run -n {shlex.quote(env_name)} {sanitized_command}"

        return await self._execute(conda_command)

    async def _handle_cd_command(self, command: str) -> CLIResult:
        """
        Handle 'cd' commands to change the current path.

        Args:
            command (str): The 'cd' command to process.

        Returns:
            TerminalOutput: The result of the 'cd' command.
        """
        try:
            parts = shlex.split(command)
            if len(parts) < 2:
                new_path = os.path.expanduser("~")
            else:
                new_path = os.path.expanduser(parts[1])

            # Handle relative paths
            if not os.path.isabs(new_path):
                new_path = os.path.join(self.current_path, new_path)

            new_path = os.path.abspath(new_path)

            if os.path.isdir(new_path):
                self.current_path = new_path
                return CLIResult(
                    output=f"Changed directory to {self.current_path}", error=""
                )
            else:
                return CLIResult(output="", error=f"No such directory: {new_path}")
        except Exception as e:
            return CLIResult(output="", error=str(e))

    @staticmethod
    def _sanitize_command(command: str) -> str:
        """
        Sanitize the command for safe execution.

        Args:
            command (str): The command to sanitize.

        Returns:
            str: The sanitized command.
        """
        # Example sanitization: restrict certain dangerous commands
        dangerous_commands = ["rm", "sudo", "shutdown", "reboot"]
        try:
            parts = shlex.split(command)
            if any(cmd in dangerous_commands for cmd in parts):
                raise ValueError("Use of dangerous commands is restricted.")
        except Exception:
            # If shlex.split fails, try basic string comparison
            if any(cmd in command for cmd in dangerous_commands):
                raise ValueError("Use of dangerous commands is restricted.")

        # Additional sanitization logic can be added here
        return command

    async def close(self):
        """Close the persistent shell process if it exists."""
        async with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                finally:
                    self.process = None

    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the asynchronous context manager and close the process."""
        await self.close()

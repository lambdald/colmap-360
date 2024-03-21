import os
import sys
import signal
import subprocess
import time
from rich import print
from misc import get_datatime_str, CONSOLE
from typing import Optional

def run_cmd(cmd_string, timeout=3600, logfile=None, working_dir="./"):
    """
    run command in subprocess with timeout constrain
    :param cmd_string:  command & parameters
    :param timeout: max time waited
    :return code: whether failed
    :return msg: message of run result
    """
    print(f"***{get_datatime_str()}***")
    print(f"RUN CMD: {cmd_string}", flush=True)
    if not logfile:
        p = subprocess.Popen(
            cmd_string,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            close_fds=True,
            start_new_session=True,
            cwd=working_dir,
        )
    else:
        p = subprocess.Popen(
            cmd_string,
            stderr=logfile,
            stdout=logfile,
            shell=True,
            close_fds=True,
            start_new_session=True,
            cwd=working_dir,
        )

    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg)
        else:
            code = 0
            msg = str(msg)
    except subprocess.TimeoutExpired:
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGUSR1)

        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"

    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)

    except KeyboardInterrupt as e:
        code = 1
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGUSR1)

        code = 1
        msg = "[ERROR]KeyboardInterrupt Error : Command '" + cmd_string
    finally:
        print(f"FINISH CMD: {cmd_string}", flush=True)
    print(f"***{get_datatime_str()}***")

    return code, msg


def run_cmd_with_log(command_string, command_name, log_dir, timeout=3600):
    """Run command in subprocess with timeout constrain and log
    Args:
    command_string: command & parameters
    command_name: name of command & logfile
    log_dir: directory of logfile
    timeout: max time waited
    """

    print(
        f"=====================================\nRUN {command_name}\ncommand: '{command_string}'\nlog: {log_dir}/{command_name}",
        flush=True,
    )
    print(f"***{get_datatime_str()}***")

    start = time.time()
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Exception occurs {e}", sys._getframe().f_code.co_name)
    with open(f"{log_dir}/{command_name}_{get_datatime_str()}.log", "w") as logfile:
        logfile.write(
            f"=====================================\nRUN {command_name}\ncommand: '{command_string}'\nlog: {log_dir}/{command_name}\n"
        )
        logfile.write(f"***{get_datatime_str()}***\n")
        logfile.flush()
        code, msg = run_cmd(cmd_string=command_string, timeout=timeout, logfile=logfile)
        if code:
            print(
                f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}", flush=True
            )
            logfile.write(f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}\n")
            print(f"***{get_datatime_str()}***")
            logfile.write(f"***{get_datatime_str()}***")
            sys.exit(1)
        else:
            print(f"--finish {command_name}, cost time: {time.time() - start} second", flush=True)
            logfile.write(f"--finish {command_name}, cost time: {time.time() - start} second\n")
            logfile.write(f"***{get_datatime_str()}***")
    return


def run_cmd_with_log_in_working_dir(command_string, command_name, log_dir, working_dir, timeout=3600):
    """Run command in subprocess with timeout constrain and log
    Args:
    command_string: command & parameters
    command_name: name of command & logfile
    log_dir: directory of logfile
    timeout: max time waited
    """

    print(
        f"=====================================\nRUN {command_name}\ncommand: '{command_string}'\nlog: {log_dir}/{command_name}",
        flush=True,
    )
    print(f"***{get_datatime_str()}***")

    start = time.time()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/{command_name}_{get_datatime_str()}.log", "w") as logfile:
        code, msg = run_cmd(cmd_string=command_string, timeout=timeout, logfile=logfile, working_dir=working_dir)
        if code:
            print(
                f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}", flush=True
            )
            logfile.write(f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}\n")
            sys.exit(1)
        else:
            print(f"--finish {command_name}, cost time: {time.time() - start} second", flush=True)
            logfile.write(f"--finish {command_name}, cost time: {time.time() - start} second\n")
    return



def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule("[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ", style="red")
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


if __name__ == "__main__":
    print(f"TEST {__file__}")
    # TEST 1
    cmd = f"nvidia-smi"
    code, msg = run_cmd(cmd, 10)
    print(f"--code: {code}\n--msg: {msg}")

    # TEST 2
    cmd = f"ls ./"
    code, msg = run_cmd(cmd, 10)
    print(f"--code: {code}\n--msg: {msg}")

import os
import sys
import signal
import subprocess
import time
from rich import print
from typing import Optional, Tuple, Callable


import datetime
import logging

def get_datatime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd_string, timeout=3600, logfile=None, working_dir="./"):
    """
    run command in subprocess with timeout constrain
    :param cmd_string:  command & parameters
    :param timeout: max time waited
    :return code: whether failed
    :return msg: message of run result
    """
    logging.info(f"RUN CMD: {cmd_string}")
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

        if msg is not None:
            msg = msg.decode("utf-8", "ignore")
        if errs is not None:
            errs = errs.decode("utf-8", "ignore")

        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error: " + str(msg) +'\n' + str(errs)
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
        logging.info(f"FINISH CMD: {cmd_string}")

    return code, msg


def run_cmd_with_log(command_string, command_name, log_dir, working_dir="./", timeout=360000, failed_callback:Optional[Callable]=None):
    """Run command in subprocess with timeout constrain and log
    Args:
    command_string: command & parameters
    command_name: name of command & logfile
    log_dir: directory of logfile
    timeout: max time waited
    """

    logging.info(
        f"=====================================\nRUN {command_name}\ncommand: '{command_string}'\nlog: {log_dir}/{command_name}",
    )

    start = time.time()
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            logging.error(f"Exception occurs {e}", sys._getframe().f_code.co_name)
    with open(f"{log_dir}/{command_name}.log", "a") as logfile:
        logfile.write(
            f"=====================================\nRUN {command_name}\ncommand: '{command_string}'\nlog: {log_dir}/{command_name}\n"
        )
        logfile.write(f"***{get_datatime_str()}***\n")
        logfile.flush()
        code, msg = run_cmd(cmd_string=command_string, timeout=timeout, logfile=logfile, working_dir=working_dir)
        if code:
            print(
                f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}", flush=True
            )
            logfile.write(f"=====================================\n{command_name}: \n-- code: {code}\n-- msg: {msg}\n")
            logfile.write(f"check the log file {log_dir}/{command_name}.log for detail.")
            logfile.write(f"***{get_datatime_str()}***")
            if failed_callback is not None:
                failed_callback(f'failed to run cmd {command_name}')
            exit(-1)
        else:
            logging.info(f"--finish {command_name}, cost time: {time.time() - start} second")
            logfile.write(f"--finish {command_name}, cost time: {time.time() - start} second\n")
            logfile.write(f"***{get_datatime_str()}***")
    return

import multiprocessing
import sys
import shutil
import os
import subprocess
import datetime
import time
import signal
import math
import glob
import logging

LOG = logging.getLogger(__name__)

# http://howto.pui.ch/post/37471155682/set-timeout-for-a-shell-command-in-python
def timeout_command(command, timeout, shell=True, stdin=None):
    """call a command and either return its output or kill it
    if it doesn't normally exit within timeout seconds and return None"""
    start = datetime.datetime.now()
    process = subprocess.Popen(command, shell=shell, stdin=subprocess.PIPE)
    if stdin:
        process.communicate(stdin)
    while process.poll() is None:
        time.sleep(0.1)
        now = datetime.datetime.now()
        if (now - start).seconds > timeout:
            os.kill(process.pid, signal.SIGTERM)
            return None

def get_merge_files_callback(base_out_file, max_size=None):
    # By default, Protocol Buffer files should not exceed 64MB otherwise the parsing will fail
    # Therefore, we support spliting the files into several parts by appending ".part-id" to the original file name

    def merge(l):

        # Process files one by one to avoid exceeding maximum length of a unix command
        part_id = 0
        out_file = base_out_file
        # os.system('rm %s' % (out_file))
        # os.system('rm %s.part-*' % (out_file))
        if os.path.isfile(out_file):
            os.remove(out_file)
        for fl in glob.glob("%s.part-*" % (out_file)):
            os.remove(fl)
        os.system('touch %s' % (out_file))

        for file in l:
            # If the file would be too big create a new one
            if max_size is not None and (os.path.getsize(out_file) + os.path.getsize(file)) >= max_size:
                out_file = base_out_file + ".part-" + str(part_id)
                os.system('touch %s' % (out_file))
                part_id += 1

                if os.path.getsize(file) >= max_size:
                    LOG.warning("File '" + file + '" bigger than the max size wont be split! Try splitting the data into smaller chunks.')

                LOG.info("Creating a new output file '" + out_file + "'")

            os.system('cat %s >> %s' % (file, out_file))

    return merge

def cbnm(task, task_callback):
    task_callback(task)

def run_tasks_no_merge(tasks, task_callback, num_threads, thread_count_divider = 2):
    p = None
    try:
        p = multiprocessing.Pool(num_threads)
        num_tasks = len(tasks)
        LOG.debug('Running %d tasks in parallel\n\n' % (num_tasks))
        done_tasks = {'num':0}

        def ProgressCallback(task):
            done_tasks['num'] += 1
            i = done_tasks['num']
            LOG.debug('\rdone {0:.2f}% [{1}/{2}]'.format((100.0 * i)/num_tasks, i, num_tasks))

        jobs = []
        for task in tasks:
            jobs.append(p.apply_async(CBNM, (task,task_callback), {}, ProgressCallback))
        for job in jobs:
            job.wait()
        sys.stderr.write('\nDone\n')

    except (KeyboardInterrupt, SystemExit):
        sys.stderr.write('Caught interrupt, terminating workers')
        p.terminate()
        p.join()
    finally:
        if p is not None:
            p.close()
            p.join()

def cbm(task, tmp_dir, task_callback):
    task_callback(task, '%s/%d' % (tmp_dir, os.getpid()))

def cbmo(tasks, tmp_dir, id, task_callback):
    for task in tasks:
        task_callback(task, '%s/%s_%d' % (tmp_dir, id, os.getpid()))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def run_ordered_tasks_and_merge_outputs(tasks, task_callback, merge_callback, num_threads, tmp_root):
    tmp_dir = os.path.join(tmp_root, 'task_dir%d' % (os.getpid()))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    p = None
    try:
        p = multiprocessing.Pool(num_threads)
        chunk_size = int(max(1, len(tasks) / num_threads / 4))
        num_tasks = math.ceil(1.0*len(tasks)/chunk_size)
        num_padding = int(math.ceil(math.log10(num_tasks)))

        LOG.debug('Running %d tasks in parallel divided in %d groups with chunk size %d\n' % (len(tasks), num_tasks, chunk_size))
        done_tasks = {'num': 0}

        def progress_callback(tasks):
            done_tasks['num'] += 1
            i = done_tasks['num']
            LOG.debug('done {0:.2f}% [{1}/{2}]'.format((100.0 * i)/num_tasks, i, num_tasks))

        jobs = []
        # Add an id to the output file that is used to order the results afterwards
        # Limit the number of files created by processing the tasks in chunks of size 'chunk_size'
        for tasks in chunks(tasks, chunk_size):
            # Produce padded id from number. e.g., 4 -> 004.
            # This is important as later on we sort using string values (i.e., file names)
            id = str(len(jobs)).zfill(num_padding)
            jobs.append(p.apply_async(cbmo, (tasks,tmp_dir,id,task_callback), {}, progress_callback))
        for job in jobs:
            job.wait()
        LOG.debug('\nDone\n')

        # Sort the results of the listdir to ensure that the files are merged in order (listdir returns files in arbitrary order)
        merge_callback(sorted([os.path.join(tmp_dir, x) for x in os.listdir(tmp_dir)]))
    except (KeyboardInterrupt, SystemExit):
        LOG.debug('Caught interrupt, terminating workers')
        p.terminate()
        p.join()
    finally:
        shutil.rmtree(tmp_dir)
        if p is not None:
            p.close()
            p.join()

def run_tasks_and_merge_outputs(tasks, task_callback, merge_callback, num_threads, tmp_dir):
    tmp_dir = os.path.join(tmp_root, 'task_dir%d' % (os.getpid()))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    p = None
    try:
        p = multiprocessing.Pool(num_threads)
        num_tasks = len(tasks)
        LOG.debug('Running %d tasks in parallel\n\n' % (num_tasks))
        done_tasks = {'num':0}

        def progress_callback(task):
            done_tasks['num'] += 1
            i = done_tasks['num']
            LOG.debug('\rdone {0:.2f}% [{1}/{2}]'.format((100.0 * i)/num_tasks, i, num_tasks))
        
        jobs = []
        for task in tasks:
            jobs.append(p.apply_async(CBM, (task,tmp_dir,task_callback), {}, progress_callback))
        for job in jobs:
            job.wait()
        LOG.debug('\nDone\n')

        merge_callback([os.path.join(tmp_dir, x) for x in os.listdir(tmp_dir)])
    except (KeyboardInterrupt, SystemExit):
        LOG.debug('Caught interrupt, terminating workers')
        p.terminate()
        p.join()
    finally:
        shutil.rmtree(tmp_dir)
        if p is not None:
            p.close()
            p.join()

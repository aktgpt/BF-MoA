import queue, threading, os, time
import shutil

fileQueue = queue.Queue()
destPath = "path/to/cop"


class ThreadedCopy:
    totalFiles = 0
    copyCount = 0
    lock = threading.Lock()

    def __init__(self, src_file_list, dest_file_list):
        # with open("filelist.txt", "r") as txt:  # txt with a file per line
        #     fileList = txt.read().splitlines()

        # if not os.path.exists(destPath):
        #     os.mkdir(destPath)

        self.totalFiles = len(src_file_list)

        print(str(self.totalFiles) + " files to copy.", flush=True)
        self.threadWorkerCopy(src_file_list, dest_file_list)

    def CopyWorker(self):
        while True:
            fileName, destFileName = fileQueue.get()
            shutil.copyfile(fileName, destFileName)
            fileQueue.task_done()
            with self.lock:
                self.copyCount += 1
                percent = (self.copyCount * 100) / self.totalFiles
                print(str(percent) + " percent copied.", flush=True)

    def threadWorkerCopy(self, fileNameList, dest_file_list):
        for i in range(16):
            t = threading.Thread(target=self.CopyWorker)
            t.daemon = True
            t.start()
        for fileName, destFileName in zip(fileNameList, dest_file_list):
            fileQueue.put((fileName, destFileName))
        fileQueue.join()


# ThreadedCopy()

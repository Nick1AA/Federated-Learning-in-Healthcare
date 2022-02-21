from distutils.dir_util import copy_tree
import os
import logging
import shutil


logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger("move_data")

args_datadir = "/tmp/CheXpert-v1.0-small/"

if __name__ == "__main__":
    logger.info("Data is on $TMP file system")
    if not os.path.isdir(args_datadir):
        logger.info("False: Copy data")
        logger.info("tmp folder exists: ")
        if os.path.isdir("/tmp/"):
            logger.info("True")
        else:
            logger.info("False")
        from_directory = "/pfs/data5/home/kit/aifb/sq8430/Federated-Learning-in-Healthcare/data/CheXpert-v1.0-small/"
        logger.info("source folder exists: ")
        if os.path.isdir(from_directory):
            logger.info("True")
        else:
            logger.info("False")
        to_directory = args_datadir
        shutil.copytree(from_directory, to_directory)
    else:
        logger.info("True")
import os
import sys

from loguru import logger

def init_log():
    logger.remove(0)
    format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|<level>{level}</level>|<cyan>{process}</cyan>:<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    logger.add(sys.stdout, format=format, enqueue=True)
    logger.add('../log/normal/log.log', format=format, rotation='1:00', enqueue=True)
    logger.add('../log/request/log.log', format=format, rotation='1:00', enqueue=True, filter=lambda x: x['extra'].get('request'))
    #创建一个日志记录命令选项
    logger.add('../log/order/log.log',format=format,rotation='1:00',enqueue=True,filter=lambda x: x['extra'].get('order'))
    logger.add('../log/error/log.log',level='ERROR', format=format, rotation='1:00', enqueue=True)
    #创建一个临时记录结果的文档用来记录时间
    logger.add('../log/result/log.log',format=format,rotation='1:00',enqueue=True,filter=lambda x: x['extra'].get('result'))








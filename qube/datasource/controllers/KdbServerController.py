import os
import platform
import subprocess
import threading

from qpython.qconnection import MessageType

from qube.utils import QubeLogger
from qube.utils.utils import is_localhost

try:
    import queue
except:
    import Queue as queue

from qpython import qconnection
from qpython.qreader import QReaderException, QException
from os.path import join, expanduser, isfile, abspath


class KdbServerController:
    """
     Class for controlling KDB/q server.
    """
    __PATH_TO_KDB = {

        'windows': [
            'c:/q/w32/q.exe', 'c:/q/w64/q.exe',
            join(expanduser('~'), 'q/w32/q.exe'), join(expanduser('~'), 'q/w64/q.exe')
        ],

        'linux': [
            join(expanduser('~'), 'q/l32/q'), join(expanduser('~'), 'q/l64/q'),
            '/usr/local/q/l32/q', '/usr/local/q/l64/q',
            '/usr/bin/q', '/usr/local/bin/q',
        ],
    }

    # or just pass props dict
    def __init__(self, host='localhost', port=5555, user='', password='', timeout=None, db_path=None, init_script=''):
        # load parameters
        self.__host = host
        self.__port = port
        self.__user = user
        self.__pass = password
        self.__timeout = timeout
        self.__logger = QubeLogger.getLogger('.'.join(['qube.datasource', 'KDB', host, str(port)]))
        self.__db_path = db_path
        self.__init_script = init_script

        self.__conn = None

        # try to connect
        self.info("Trying to connect to %s:%d ..." % (self.__host, self.__port))
        if not self.__connect_and_run_if_need():
            raise ConnectionAbortedError("Can't setup connection")

        if not self.__check_if_kdb_alive():
            raise ConnectionAbortedError("Can't setup connection")

        self.info("connected")

    @staticmethod
    def create(kdb_ds_props):
        port = kdb_ds_props.get('ports_from') if kdb_ds_props.get('ports_from') else kdb_ds_props.get('port')
        return KdbServerController(port=port, host=kdb_ds_props.get('host'), timeout=kdb_ds_props.get('timeout'))

    def __connect(self):
        if not self.__is_connected():
            self.__conn = qconnection.QConnection(host=self.__host, port=self.__port, pandas=True,
                                                  username=self.__user, password=self.__pass, timeout=self.__timeout)
            try:
                self.__conn.open()
            except ConnectionRefusedError as e:
                self.__conn = None
                return False
        return True

    def exec(self, query, skip_connection_check=False):
        """
        Interface method to be called outside. It check if server is alive and starts it if need

        :param query: q query to be executed
        :skip_connection_check: set true for skipping connection check
        :return: query resutl
        """
        if not skip_connection_check:
            if not self.__check_if_kdb_alive():
                if not self.__connect_and_run_if_need():
                    raise ConnectionAbortedError("Can't setup connection")
        return self.__conn(query)

    def reload(self):
        self.__conn('.Q.l[`.]')

    def __connect_and_run_if_need(self):
        # try to connect
        if not self.__connect():
            self.info("KDB server is not running on %s:%d" % (self.__host, self.__port))
            # if unable to connect and it's local DB try to run it
            if is_localhost(self.__host):
                if self.__run_server():
                    # if server running properly try to connect
                    if not self.__connect():
                        self.error("KDB server is started but can't connect to it !")
                        return False
                else:
                    self.error("Unable to start KDB server")
                    return False
            else:
                self.error("Don't know how to run KDB server on remote host !")
                return False

        # run initialization script if set
        if self.__init_script:
            self.info("Run initialization script from '%s'" % self.__init_script)
            try:
                if self.__db_path:
                    self.__conn('db_path:"%s"' % self.__db_path)
                self.__conn('system["l %s"]' % self.__init_script)
            except QException as q_err:
                self.error("Exception in initialization script from '%s'" % self.__init_script)
                self.shutdown()
                raise q_err

            self.info("Done")

        return True

    def __is_connected(self):
        return self.__conn and self.__conn.is_connected()

    def __check_if_kdb_alive(self):
        if not self.__is_connected():
            self.__connect()
        try:
            return bool(self.__conn('.z.K'))
        except QReaderException as e:
            self.__conn = None
            self.error("Error reading from kdb server at %s:%d : '%s'" % (self.__host, self.__port, e))
        return False

    def __run_server(self):
        self.info('Attemp to run kdb server ...')
        run_os = platform.system().lower()
        if run_os in self.__PATH_TO_KDB.keys():
            q_exec = list(filter(lambda x: isfile(x) and os.access(x, os.X_OK), self.__PATH_TO_KDB[run_os]))
            if not q_exec:
                self.error("KDB server executable not found !")
                return False

            # run first found
            self.info("Found kdb executable files : " + repr(q_exec) + " running from '%s' " % q_exec[0])

            p = subprocess.Popen([q_exec[0], "-p", str(self.__port)], stdout=subprocess.PIPE, bufsize=1)
            self.__get_kdb_process_echo(p)
            return True
        else:
            self.error("Don't know how to run kdb in '%s' operating system !" % run_os)

        return False

    def __get_kdb_process_echo(self, p):
        def enqueue_output(out, queue):
            for line in iter(out.readline, b''):
                queue.put(line)
            out.close()

        q = queue.Queue()
        t = threading.Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True  # thread dies with the program
        t.start()

        attempt = 0
        while True:
            try:
                attempt += 1
                line = q.get(timeout=.01)
            except queue.Empty:
                if attempt >= 50:
                    raise ConnectionError('Awaiting too long to get Echo message from kdb server process')
            else:  # got line
                self.info("Got response from kdb server: " + str(line))
                break

    def shutdown(self):
        """
        Shutdown kdb server
        """
        if self.__check_if_kdb_alive():
            # calling instead of self.__conn.async("\\\\"). async is reserved key now
            self.__conn.query(MessageType.ASYNC, "\\\\")
            self.info("Server is stopped")
        else:
            self.info("Server is already stopped")

    def warn(self, msg):
        self.__logger.warning(msg)

    def info(self, msg):
        self.__logger.info(msg)

    def error(self, msg):
        self.__logger.error(msg)

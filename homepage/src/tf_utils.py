
import tensorflow as tf

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class SessionHandler(Singleton):
    def set_default(self):
        self.is_session_init = 1
        self.session = tf.InteractiveSession()


    def get_session(self):
        # if self.session is not None:
        #     self.session = tf.InteractiveSession()
        return self.session

    # initialize the variables have not been initialized
    def initialize_variables(self):
        v_list = tf.all_variables()
        for v in v_list:
            if not tf.is_variable_initialized(v).eval():
                self.session.run(v.initialized_value())

    def re_set(self):
        self.session.close()
        del self.session
        self.set_default()
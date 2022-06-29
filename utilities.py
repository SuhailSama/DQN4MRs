

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import os 

def movement_animation(episode_i,all_state_buffer,all_step_reward_buffer,all_step_action_buffer,goal,observation_method):
    grid_SIZE = 10
    fig, ax = plt.subplots()
    ax.plot([goal.x],[goal.y],'go')
    ax.axis([0,grid_SIZE,0,grid_SIZE])
    l, = ax.plot([],[],'ro')
    def animate(step_i):
        if step_i > 0:
            del fig.texts[0:len(fig.texts)]
        ax.plot([goal.x],[goal.y],'go')
        if observation_method == 0:
            l.set_data(all_state_buffer[step_i-1][0,0,:], all_state_buffer[step_i-1][0,1,:])
        elif observation_method == 1:
            l.set_data(goal.x-all_state_buffer[step_i-1][0,0,:], goal.y-all_state_buffer[step_i-1][0,1,:])
        fig.text(0.1, 0.9, 'Step Reward:'+str(all_step_reward_buffer[step_i-1]), size=10, color='purple')
        fig.text(0.5, 0.9, 'action:'+str(all_step_action_buffer[step_i-1]), size=10, color='purple')
    ani = animation.FuncAnimation(fig, animate, frames=len(all_state_buffer))
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())
    f = r"animation/cell_movement_animation"+str(episode_i)+".gif" 
    writergif = animation.PillowWriter(fps=4) 
    ani.save(f, writer=writergif)


# Own Tensorboard class

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)#tf.summary.FileWriter(self.log_dir)
        self._log_write_dir = self.log_dir
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    def on_train_batch_end(self, batch, logs=None):
        pass
    def _write_logs(self, logs, index):
        
        writer = tf.summary.create_file_writer("self.log_dir")
        with writer.as_default():
          for step in range(100):
            # other model code would go here
            tf.summary.scalar("my_metric", 0.5, step=step)
            writer.flush()

    # Custom method for  saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


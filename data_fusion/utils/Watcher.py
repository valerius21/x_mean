from IPython.core.display import HTML
from matplotlib import animation, pyplot as plt
from matplotlib.patches import Rectangle

from data_fusion.utils.data_parsing import result


class Watcher:
    plt = None
    result = []
    xlim = (300, 500)
    ylim = (1000, 1300)
    ani = None

    def __init__(self, plt, result):
        self.plt = plt
        self.plt.style.use('dark_background')
        self.result = result

    def show(self):
        """
        Draws the animation
        """
        self.ani = animation.FuncAnimation(self.plt.gcf(), self.step, frames=len(self.result), interval=100)
        self.plt.show()
        return HTML(self.ani.to_html5_video())

    def step(self, frame):
        """
        stepper for the animation
        """

        row = self.result[frame]

        # extract values from row
        x, y = row[0], row[1]
        vx, vy = row[3], row[4]
        radar_ego_x = row[6][0]
        radar_ego_y = row[6][1]

        # clear plot + set limits
        self.plt.cla()
        self.plt.xlim(self.xlim[0], self.xlim[1])
        self.plt.ylim(self.ylim[0], self.ylim[1])

        # Get vehicle annotation from sample
        vehicles = get_vehicles_from_sample(scene_anns[frame])
        vehicle_coordinates = [v.get_trans() for v in vehicles]
        vehicle_x = [p[0] for p in vehicle_coordinates]
        vehicle_y = [p[1] for p in vehicle_coordinates]

        # draw radar points
        self.plt.scatter(x, y, s=1)

        # draw ego post
        self.plt.scatter(radar_ego_x, radar_ego_y, s=50, color='red')

        # draw vehicle annoations
        color = 'yellow'
        for i in range(len(vehicle_y)):
            self.plt.gca().add_patch(
                Rectangle(vehicle_coordinates[i], vehicles[i].w, vehicles[i].h, color=color, fill=False)
            )

        self.plt.plot()


w = Watcher(plt=plt, result=result)
w.show()

import pickle
import time

from manim import *


class twoD_oscillator2(ThreeDScene):

    def V(self, x, y):
        # return np.sin(0.5*x) + np.sin(y) + np.sin(x) + 0.05*x**2 + 0.03*y**2
        return 0.5*(x-1)**2
    
    def parabolic(self, x):
        return (x, 0, 0.5*(x-1)**2)

    def acc(self, x, y, delta=0.0001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
        start_time = time.time()
        min_x, max_x = x_axis[0], x_axis[-1]
        dt = t_axis[1] - t_axis[0]
        pos_list = [(x0, y0)]
        vel_list = [(0, 0)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0], pos_list[i-1][1])
            vel_curr = (vel_list[i-1][0] + acc_prev[0]*dt, vel_list[i-1][1] + acc_prev[1]*dt)
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, max_x), min_x), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, max_x), min_x))

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        end_time = time.time()
        print(f"generate_pos_list: {end_time - start_time} sec")
        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        # print(f"pos = {(x, y, self.V(x, y))}")
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES)

        x_axis = np.arange(-10, 10, 0.001)
        t_axis = np.arange(0, 50, 0.00001)
        initial_pos = (0, 0)
        self.generate_pos_list(initial_pos[0], initial_pos[1], t_axis, x_axis)
        i = ValueTracker(0)
        ball_radius = 0.1

        ball_pos = always_redraw(
            lambda: Sphere(
                self.ball_path_func(i.get_value()),
                radius=ball_radius,
                color=RED
            )
        )
        ball_height = always_redraw(
            lambda: DecimalNumber(
                self.ball_path_func(i.get_value())[2]
            )
        )
        axes = ThreeDAxes(
            x_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
            y_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
            z_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
        )
        surface = axes.plot_surface(
            self.V,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(axes, surface, ball_pos)

        # self.move_camera(phi=75 * DEGREES)
        self.move_camera(phi=90 * DEGREES, theta=-90*DEGREES)

        self.add(ball_height)

        ball_path = ParametricFunction(self.ball_path_func, t_range=[0, len(t_axis)-1, 1], color=RED)

        self.add(ball_path)

        parabolic_path = ParametricFunction(self.parabolic, t_range=[0, 2, 0.001], color=GREEN)

        self.add(parabolic_path)

        # self.begin_ambient_camera_rotation(
        #     rate=PI / 10, about="theta"
        # )  # Rotates at a rate of radians per second

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=10)

        
        self.wait(2)
        # self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")


class EnergyLandscape(ThreeDScene):

    def V(self, x, y):
        return np.sin(0.5*x) + np.sin(y) + np.sin(x) + 0.05*x**2 + 0.03*y**2

    def acc(self, x, y, delta=0.0001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
        start_time = time.time()
        min_x, max_x = x_axis[0], x_axis[-1]
        dt = t_axis[1] - t_axis[0]
        pos_list = [(x0, y0)]
        vel_list = [(0, 0)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0], pos_list[i-1][1])
            vel_curr = (vel_list[i-1][0] + acc_prev[0]*dt, vel_list[i-1][1] + acc_prev[1]*dt)
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, max_x), min_x), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, max_x), min_x))

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        end_time = time.time()
        print(f"generate_pos_list: {end_time - start_time} sec")
        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        # print(f"pos = {(x, y, self.V(x, y))}")
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES)

        x_axis = np.arange(-10, 10, 0.001)
        t_axis = np.arange(0, 50, 0.00001)
        initial_pos = (0, 0)

        axes = ThreeDAxes(
            x_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
            y_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
            z_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
        )
        surface = axes.plot_surface(
            self.V,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(axes, surface)

        self.move_camera(phi=75 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        
        self.wait(50)
        print("waited 50")
        self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")


class AnalogComputing(Scene):
    def construct(self):
        disc_radius = 1
        disc1 = Circle(disc_radius).set_fill(opacity=1).move_to(3*LEFT + 2*DOWN)
        self.add(disc1)
        disc1_horizonLine = Line(disc1.get_center(), end=disc1.get_center()+disc_radius*RIGHT)
        disc1_rotateLine = Line(disc1.get_center(), end=disc1.get_center()+disc_radius*RIGHT)
        self.add(disc1_horizonLine, disc1_rotateLine)
        self.wait(1)


        rad_1 = 0.6412
        self.play(
            Rotate(
                disc1_rotateLine,
                angle=rad_1,
                about_point=disc1_rotateLine.start,
                rate_func=linear
            )
        )
        arc_1 = Arc(radius=0.5, angle=rad_1, color=WHITE).move_arc_center_to(disc1.get_center())
        angle_1 = DecimalNumber(rad_1, num_decimal_places=4, font_size=20, color=WHITE).move_to(arc_1.get_center() + 0.4*RIGHT)
        self.add(arc_1, angle_1)


        # self.play(
        #     Rotate(
        #         Square(side_length=0.5).shift(UP * 2),
        #         angle=2*PI,
        #         about_point=ORIGIN,
        #         rate_func=linear,
        #     ),
        #     Rotate(Square(side_length=0.5), angle=2*PI, rate_func=linear),
        #     )
        self.wait()


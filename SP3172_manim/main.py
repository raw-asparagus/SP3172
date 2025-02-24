import pickle
import time

from manim import *


class twoD_oscillator3(ThreeDScene):

    def V(self, x, y):
        return 0.05 * (x**2 + y**2)
    
    def V_surface(self, x, y):
        return (x, y, self.V(x, y))

    def acc(self, x, y, delta=0.0001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
        decay = 0.3

        start_time = time.time()
        min_x, max_x = x_axis[0], x_axis[-1]
        dt = t_axis[1] - t_axis[0]
        pos_list = [(x0, y0)]
        vel_list = [(0, 0)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0], pos_list[i-1][1])
            vel_curr = ((1-decay)*vel_list[i-1][0] + acc_prev[0]*dt, (1-decay)*vel_list[i-1][1] + acc_prev[1]*dt)
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, max_x), min_x), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, max_x), min_x))

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        end_time = time.time()
        print(f"generate_pos_list: {end_time - start_time} sec")
        
        # file = open("2dSHO_pos_list_Decay", "rb")
        # pos_list = pickle.load(file)
        file = open("2dSHO_pos_list_Decay", "wb")
        pickle.dump(pos_list, file)
        file.close()

        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        # print(f"pos = {(x, y, self.V(x, y))}")
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES, zoom=0.25)

        x_axis = np.arange(-10, 10, 0.001)
        t_axis = np.arange(0, 50, 0.00001)
        initial_pos = (1, 1)
        self.generate_pos_list(initial_pos[0], initial_pos[1], t_axis, x_axis)
        i = ValueTracker(0)
        ball_radius = 0.2

        ball_pos = always_redraw(
            lambda: Sphere(
                self.ball_path_func(i.get_value()),
                radius=ball_radius,
                color=RED,
            ).set_opacity(1)
        )
        surface = Surface(
            self.V_surface,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(surface, ball_pos)

        self.wait()

        self.move_camera(phi=45 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        self.wait(2)

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=5)

        self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")



class EnergyLandscape2(ThreeDScene):

    def V(self, x, y):
        return np.sin(0.5*x) + np.sin(y) + np.sin(x) + 0.05*x**2 + 0.03*y**2
    
    def V_surface(self, x, y):
        return (x, y, self.V(x, y))
    
    def parabolic(self, x):
        return (x, 0, 0.5*(x-1)**2)

    def acc(self, x, y, delta=0.0001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
        decay = 0.0

        start_time = time.time()
        min_x, max_x = x_axis[0], x_axis[-1]
        dt = t_axis[1] - t_axis[0]
        pos_list = [(x0, y0)]
        vel_list = [(0, 0)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0], pos_list[i-1][1])
            vel_curr = ((1-decay)*vel_list[i-1][0] + acc_prev[0]*dt, (1-decay)*vel_list[i-1][1] + acc_prev[1]*dt)
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, max_x), min_x), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, max_x), min_x))

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        end_time = time.time()
        print(f"generate_pos_list: {end_time - start_time} sec")
        
        # file = open("pos_list_Decay", "rb")
        # pos_list = pickle.load(file)
        file = open("pos_list_Decay", "wb")
        pickle.dump(pos_list, file)
        file.close()

        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        # print(f"pos = {(x, y, self.V(x, y))}")
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES, zoom=0.25)

        x_axis = np.arange(-10, 10, 0.001)
        t_axis = np.arange(0, 50, 0.00001)
        initial_pos = (0, 0)
        self.generate_pos_list(initial_pos[0], initial_pos[1], t_axis, x_axis)
        i = ValueTracker(0)
        ball_radius = 0.2

        ball_pos = always_redraw(
            lambda: Sphere(
                self.ball_path_func(i.get_value()),
                radius=ball_radius,
                color=RED,
            ).set_opacity(1)
        )
        surface = Surface(
            self.V_surface,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(surface, ball_pos)

        self.wait()

        self.move_camera(phi=45 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        self.wait(2)

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=5)

        self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")


class Annealing2(ThreeDScene):

    def V0(self, x, y):
        return 0.5 * (x**2 + y**2)

    def Vp(self, x, y):
        return np.sin(0.5*x) + np.sin(y) + np.sin(x) + 0.05*x**2 + 0.03*y**2
    
    def V(self, x, y, t):
        return (1-t)*self.V0(x, y) + t*self.Vp(x, y)

    def V_surface(self, x, y, t):
        return (x, y, self.V(x, y, t))

#  come back later
    def acc(self, x, y, t, delta=0.0001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
        decay = 0.0

        start_time = time.time()
        min_x, max_x = x_axis[0], x_axis[-1]
        dt = t_axis[1] - t_axis[0]
        pos_list = [(x0, y0)]
        vel_list = [(0, 0)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0], pos_list[i-1][1])
            vel_curr = ((1-decay)*vel_list[i-1][0] + acc_prev[0]*dt, (1-decay)*vel_list[i-1][1] + acc_prev[1]*dt)
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, max_x), min_x), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, max_x), min_x))

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        end_time = time.time()
        print(f"generate_pos_list: {end_time - start_time} sec")
        
        # file = open("pos_list_Decay", "rb")
        # pos_list = pickle.load(file)
        file = open("pos_list_Decay", "wb")
        pickle.dump(pos_list, file)
        file.close()

        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        # print(f"pos = {(x, y, self.V(x, y))}")
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES, zoom=0.25)

        x_axis = np.arange(-10, 10, 0.001)
        t_axis = np.arange(0, 50, 0.00001)
        initial_pos = (0, 0)
        self.generate_pos_list(initial_pos[0], initial_pos[1], t_axis, x_axis)
        i = ValueTracker(0)
        ball_radius = 0.2

        ball_pos = always_redraw(
            lambda: Sphere(
                self.ball_path_func(i.get_value()),
                radius=ball_radius,
                color=RED,
            ).set_opacity(1)
        )
        surface = always_redraw(
            lambda: Surface(
            self.V_surface,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )
        )
        surface = Surface(
            self.V_surface,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(surface, ball_pos)

        self.wait()

        self.move_camera(phi=45 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        self.wait(2)

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=5)

        self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")



class AnalogComputing(Scene):
    def construct(self):
        disc_radius = 1

        rad_1 = 0.6412
        disc1 = Circle(disc_radius).set_fill(opacity=0.9).move_to(1*LEFT + 1*DOWN)
        self.add(disc1)
        disc1_horizonLine = Line(disc1.get_center(), end=disc1.get_center()+disc_radius*RIGHT)
        disc1_rotateLine = Line(disc1.get_center(), end=disc1.get_center()+disc_radius*RIGHT)
        self.add(disc1_horizonLine, disc1_rotateLine)
        self.wait(5)
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
        self.wait(5)

        rad_2 = 0.3459
        disc2 = Circle(disc_radius).set_fill(opacity=0.9).move_to(disc1_rotateLine.get_end()).set_z_index(-1)
        self.add(disc2)
        disc2_horizonLine = Line(disc2.get_center(), end=disc2.get_center()+disc_radius*RIGHT)
        disc2_rotateLine = Line(disc2.get_center(), end=disc2.get_center()+disc_radius*RIGHT)
        self.add(disc2_horizonLine, disc2_rotateLine)
        self.wait(1)
        self.play(
            Rotate(
                disc2_rotateLine,
                angle=rad_2,
                about_point=disc2_rotateLine.start,
                rate_func=linear
            )
        )
        arc_2 = Arc(radius=0.5, angle=rad_2, color=WHITE).move_arc_center_to(disc2.get_center())
        angle_2 = DecimalNumber(rad_2, num_decimal_places=4, font_size=20, color=WHITE).move_to(arc_2.get_center() + 0.4*RIGHT)
        self.add(arc_2, angle_2)
        self.wait(3)

        rad_3 = 1.207
        disc3 = Circle(disc_radius).set_fill(opacity=0.9).move_to(disc2_rotateLine.get_end()).set_z_index(-1)
        self.add(disc3)
        disc3_horizonLine = Line(disc3.get_center(), end=disc3.get_center()+disc_radius*RIGHT)
        disc3_rotateLine = Line(disc3.get_center(), end=disc3.get_center()+disc_radius*RIGHT)
        self.add(disc3_horizonLine, disc3_rotateLine)
        self.wait(1)
        self.play(
            Rotate(
                disc3_rotateLine,
                angle=rad_3,
                about_point=disc3_rotateLine.start,
                rate_func=linear
            )
        )
        arc_3 = Arc(radius=0.5, angle=rad_3, color=WHITE).move_arc_center_to(disc3.get_center())
        angle_3 = DecimalNumber(rad_3, num_decimal_places=4, font_size=20, color=WHITE).move_to(arc_3.get_center() + 0.4*RIGHT)
        self.add(arc_3, angle_3)
        self.wait(3)

        
        self.wait()


from manim import *


class twoD_oscillator2(ThreeDScene):

    def V(self, x, y):
        # return np.sin(0.5*x) + np.sin(y) + np.sin(x) + 0.05*x**2 + 0.03*y**2
        return 0.5*(x-1)**2
    
    def V_surface(self, x, y):
        return (x, y, self.V(x, y))
    
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
        # axes = ThreeDAxes(
        #     x_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
        #     y_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
        #     z_range=[round(x_axis[0]), round(x_axis[-1]), round((x_axis[-1]-x_axis[0])/20)],
        # )
        # surface = axes.plot_surface(
        #     self.V,
        #     u_range=[x_axis[0], x_axis[-1]],
        #     v_range=[x_axis[0], x_axis[-1]],
        #     color=BLUE
        # )
        print("everything fine")
        surface = Surface(
            self.V_surface,
            u_range=[x_axis[0], x_axis[-1]],
            v_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )
        print("still fine?")

        # self.add(axes, surface, ball_pos)
        self.add(surface, ball_pos)
        print("hmmm.....")

        # self.move_camera(phi=75 * DEGREES)
        self.move_camera(phi=90 * DEGREES, theta=-90*DEGREES)

        self.add(ball_height)

        # ball_path = axes.plot_parametric_curve(self.ball_path_func, t_range=[0, len(t_axis)-1, 1], color=RED)

        # ball_path = ParametricFunction(self.ball_path_func, t_range=[0, len(t_axis)-1, 1], color=RED)

        # self.add(ball_path)

        parabolic_path = ParametricFunction(self.parabolic, t_range=[0, 2, 0.001], color=GREEN)

        self.add(parabolic_path)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=5)

        
        self.wait(2)
        self.stop_ambient_camera_rotation()

        self.wait()
        print("waited")

class twoD_oscillator1(ThreeDScene):

    def V(self, x, y):
        return 0.25 * (x**2 + y**2)

    def acc(self, x, y, delta=0.001):
        dV_dx = 0.5 * (self.V(x + delta, y) - self.V(x - delta, y)) / delta
        dV_dy = 0.5 * (self.V(x, y + delta) - self.V(x, y - delta)) / delta
        return (-dV_dx, -dV_dy)

    def generate_pos_list(self, x0, y0, t_axis, x_axis):
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
        
        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        # print(f"ball_path_func i = {i}, int(i) = {int(i)}")
        i = int(i)
        x, y = self.pos_list[i]
        return (x, y, self.V(x, y))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES)

        x_axis = np.arange(-10, 10, 0.01)
        t_axis = np.arange(0, 50, 0.001)
        initial_pos = (1, 1)
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

        self.move_camera(phi=75 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=10)

        
        self.wait(2)
        self.stop_ambient_camera_rotation()

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


class RollingBall(ThreeDScene):
    def construct(self):

        self.set_camera_orientation(phi=0 * DEGREES, theta=0* DEGREES)
        axes = ThreeDAxes(
            x_range=[-100, 100, 0.1],
            y_range=[-100, 100, 0.1],
            z_range=[-6, 6, 0.1],
            # x_length=8,
            # y_length=6,
            # z_length=6,
        )
        
        surface = axes.plot_surface(
            V,
            u_range=[-2 * PI, 2 * PI],
            v_range=[-2 * PI, 2 * PI],
            color=RED
        )

        self.add(axes, surface)

        ball_radius = 0.1
        dt = 0.001
        t_max = 100
        ball_path = generate_path(0.5, 0.5, dt=dt, t_max=t_max)

        def ball_pos(t):
            nonlocal ball_path, dt, ball_radius
            i = min(int(t // dt), len(ball_path)-1)
            print(f"t: {t}, i: {i}")
            x, y = ball_path[i]
            return (x, y, V(x, y) + ball_radius)
        
        ball_path = ParametricFunction(ball_pos, t_range=(0, t_max), color=RED)
        self.add(ball_path, axes)

        sphere = Sphere(radius=ball_radius,color=RED)
        self.add(sphere, axes)

        self.move_camera(phi=75 * DEGREES)
        self.move_camera(theta=75 * DEGREES)
        self.wait()
        
        self.play(MoveAlongPath(sphere, ball_path), run_time=10, rate_func=linear)

        self.wait()

def V(x, y):
    return 0.25 * (x**2 + y**2)
def V(x, y):
    return 0.5*np.sin(0.1*x*y) + 0.5*np.sin(0.2*x*y) + 0.5*np.sin(0.1*x + 0.2*y)

def neg_grad(V, x, y, delta=0.001):
    """ V is the potential function V(x, y)"""
    dV_dx = 0.5 * (V(x + delta, y) - V(x - delta, y)) / delta
    dV_dy = 0.5 * (V(x, y + delta) - V(x, y - delta)) / delta
    return (-dV_dx, -dV_dy)

def generate_path(x_0, y_0, dt=0.1, t_max=1):
    pos_list = [(x_0, y_0)]
    vel_list = [(0, 0)]
    for i in range(1, int(t_max//dt)):
        acc_prev = neg_grad(V, pos_list[i-1][0], pos_list[i-1][1])
        vel_curr = (vel_list[i-1][0] + acc_prev[0]*dt, vel_list[i-1][1] + acc_prev[1]*dt)
        pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*dt, 100), -100), max(min(pos_list[i-1][1] + vel_list[i-1][1]*dt, 100), -100))

        pos_list.append(pos_curr)
        vel_list.append(vel_curr)
    return pos_list


class oneD_oscillator1(ThreeDScene):

    def V(self, x):
        return 0.25 * x**2

    def acc(self, x, delta=0.001):
        dV_dx = 0.5 * (self.V(x + delta) - self.V(x - delta)) / delta
        return (-dV_dx, )

    def generate_pos_list(self, x0, t_axis, x_axis):
        min_x, max_x = x_axis[0], x_axis[-1]
        pos_list = [(x0,)]
        vel_list = [(0,)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0])
            vel_curr = (vel_list[i-1][0] + acc_prev[0]*(t_axis[i] - t_axis[i-1]), )
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*(t_axis[i] - t_axis[i-1]), max_x), min_x), )

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        i = int(i)
        x = self.pos_list[i][0]
        # print(f"x: {x}, V:{self.V(x)}")
        return (x, 0, self.V(x))

    def construct(self):
        x_axis = np.arange(-10, 10, 0.01)
        t_axis = np.arange(0, 50, 0.001)
        initial_pos = (1,)
        self.generate_pos_list(initial_pos[0], t_axis, x_axis)

        self.ball_radius = 0.1
        sphere = Sphere((initial_pos[0], 0, self.V(initial_pos[0])), radius=self.ball_radius, color=RED)
        self.add(sphere)
        # ball_path = ParametricFunction(
        #     lambda t: ( pos_list[int(t)][0], 0, self.V(pos_list[int(t)][0] + ball_radius) ),
        #     t_range=[t_axis[0], t_axis[-1], (t_axis[1]-t_axis[0])],
        # )
        # ball_path = ParametricFunction(self.ball_path_func, t_range=[0, len(t_axis)-1, 1])
        ball_path = always_redraw(self.ball_path_func)

        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES)
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
        )
        self.add(axes)

        curve = axes.plot_parametric_curve(
            lambda t: (t, 0, self.V(t)),
            t_range=[x_axis[0], x_axis[-1]],
            color=RED
        )
        self.add(curve)

        self.move_camera(phi=75 * DEGREES)

        self.play(MoveAlongPath(sphere, ball_path), run_time=10, rate_func=linear)

        self.wait()
        print("wait")



class oneD_oscillator2(ThreeDScene):

    def V(self, x):
        return 0.25 * x**2

    def acc(self, x, delta=0.001):
        dV_dx = 0.5 * (self.V(x + delta) - self.V(x - delta)) / delta
        return (-dV_dx, )

    def generate_pos_list(self, x0, t_axis, x_axis):
        min_x, max_x = x_axis[0], x_axis[-1]
        pos_list = [(x0,)]
        vel_list = [(0,)]
        for i in range(1, len(t_axis)):
            acc_prev = self.acc(pos_list[i-1][0])
            vel_curr = (vel_list[i-1][0] + acc_prev[0]*(t_axis[i] - t_axis[i-1]), )
            pos_curr = (max(min(pos_list[i-1][0] + vel_list[i-1][0]*(t_axis[i] - t_axis[i-1]), max_x), min_x), )

            pos_list.append(pos_curr)
            vel_list.append(vel_curr)
        
        self.pos_list = pos_list
        return pos_list
    
    def ball_path_func(self, i):
        print(f"ball_path_func i = {i}")
        i = int(i)
        x = self.pos_list[i][0]
        return (x, 0, self.V(x))

    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90* DEGREES)

        x_axis = np.arange(-10, 10, 0.01)
        t_axis = np.arange(0, 50, 0.001)
        initial_pos = (1,)
        self.generate_pos_list(initial_pos[0], t_axis, x_axis)
        i = ValueTracker(0)
        ball_radius = 0.05

        ball_pos = always_redraw(
            lambda: Sphere(
                self.ball_path_func(i.get_value()),
                radius=ball_radius,
                color=RED
            )
        )
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
        )
        curve = axes.plot_parametric_curve(
            lambda t: (t, 0, self.V(t)),
            t_range=[x_axis[0], x_axis[-1]],
            color=BLUE
        )

        self.add(axes, curve, ball_pos)

        self.move_camera(phi=75 * DEGREES)

        self.play(i.animate.set_value(len(t_axis)-1), rate_func=linear, run_time=10)

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



class MovingCameraTemplate(MovingCameraScene):
    def construct(self):
        text = Text("Hello World").set_color(BLUE)
        self.add(text)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=text.width * 1.2))
        self.wait(0.3)
        self.play(Restore(self.camera.frame))

class TangentAnimation(Scene):
    def construct(self):
        ax = Axes()
        sine = ax.plot(np.sin, color=RED)
        alpha = ValueTracker(0)
        point = always_redraw(
            lambda: Dot(
                sine.point_from_proportion((alpha.get_value())**2),
                color=BLUE
            )
        )
        self.add(ax, sine, point)
        self.play(alpha.animate.set_value(1), rate_func=linear, run_time=2)


class Annealing(ThreeDScene):

    def construct(self):
        axes = ThreeDAxes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            z_range=[-6, 6, 1],
            x_length=8,
            y_length=6,
            z_length=6,
        )
        
        def V(x, y):
            return 0.1 * (x**2 + y**2)
        
        surface = axes.plot_surface(
            V,
            u_range=[-2 * PI, 2 * PI],
            v_range=[-2 * PI, 2 * PI],
            color=RED
        )

        self.add(axes, surface)
        self.move_camera(phi=75 * DEGREES)
        self.move_camera(theta=75 * DEGREES)
        self.wait()

        ball_radius = 0.3
        ball = Dot(radius=0.3)
        def ball_function_path(t):
            return t
        
        ball.move_to(ball_path.points[0])

        ##THE CAMERA IS AUTO SET TO PHI = 0 and THETA = -90

        self.move_camera(phi=60 * DEGREES)
        self.wait()
        self.move_camera(theta=-45 * DEGREES)

        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="theta"
        )  # Rotates at a rate of radians per second
        self.wait()
        self.play(Create(rects), run_time=3)
        self.play(Create(graph2))
        self.wait()
        self.stop_ambient_camera_rotation()

        self.wait()
        self.begin_ambient_camera_rotation(
            rate=PI / 10, about="phi"
        )  # Rotates at a rate of radians per second
        self.wait(2)
        self.stop_ambient_camera_rotation()

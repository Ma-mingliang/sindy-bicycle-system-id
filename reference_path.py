"""Reference path geometry for Stage 2 path tracking.

Complex path with 6 segments (matching HRRL):
  1. Straight along +X (y=0)
  2. Left arc, center (55,15), radius 15
  3. Straight along +Y (x=70)
  4. Left arc, center (55,35), radius 15
  5. Right arc, center (28,35), radius 12
  6. Straight along -X (y=23)
"""

import numpy as np
import math


class ReferencePath:
    """Complex reference path for bicycle path tracking."""

    def get_closest_point(self, x, y, bike_heading):
        """Compute path tracking errors at position (x, y).

        Args:
            x, y: front wheel position in world frame
            bike_heading: bike direction angle (rad from +X axis)

        Returns:
            dict with lateral_error, course_error_angle, curvature, segment_id
        """
        # Segment 1: Straight along +X, y=0, x < 55
        if x < 55 and y < 5:
            lateral_error = y
            course_error_angle = self._heading_error_straight(bike_heading, 0.0)
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': 0.0, 'segment': 1}

        # Segment 2: Left arc, center (55,15), radius 15
        if 55 < x < 75 and -5 < y < 15:
            dx, dy = 55 - x, 15 - y
            dist = math.sqrt(dx*dx + dy*dy)
            lateral_error = 15 - dist
            course_error_angle = self._heading_error_arc(bike_heading, dx, dy, 'left')
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': 1.0/15, 'segment': 2}

        # Segment 3: Straight along +Y, x=70
        if 65 < x < 75 and 15 < y < 35:
            lateral_error = -(x - 70)
            course_error_angle = self._heading_error_straight(bike_heading, math.pi/2)
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': 0.0, 'segment': 3}

        # Segment 4: Left arc, center (55,35), radius 15
        if 35 < x < 75 and 35 < y < 55:
            dx, dy = 55 - x, 35 - y
            dist = math.sqrt(dx*dx + dy*dy)
            lateral_error = 15 - dist
            course_error_angle = self._heading_error_arc(bike_heading, dx, dy, 'left')
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': 1.0/15, 'segment': 4}

        # Segment 5: Right arc, center (28,35), radius 12
        if 28 < x < 45 and 18 < y < 35:
            dx, dy = 28 - x, 35 - y
            dist = math.sqrt(dx*dx + dy*dy)
            lateral_error = dist - 12
            course_error_angle = self._heading_error_arc(bike_heading, dx, dy, 'right')
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': -1.0/12, 'segment': 5}

        # Segment 6: Straight along -X, y=23
        if 0 < x < 28 and 15 < y < 58:
            lateral_error = 23 - y
            course_error_angle = self._heading_error_straight(bike_heading, math.pi)
            return {'lateral_error': lateral_error, 'course_error_angle': course_error_angle,
                    'curvature': 0.0, 'segment': 6}

        # Default: not on any segment
        return {'lateral_error': 0.0, 'course_error_angle': 0.0,
                'curvature': 0.0, 'segment': 0}

    def _heading_error_straight(self, bike_heading, path_heading):
        """Course error angle for straight segment."""
        error = bike_heading - path_heading
        # Normalize to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        return error

    def _heading_error_arc(self, bike_heading, dx, dy, turn_direction):
        """Course error angle for arc segment.

        dx, dy: vector from position to arc center
        """
        # Radial direction (from position to center)
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-6:
            return 0.0

        # Tangent direction (perpendicular to radial)
        if turn_direction == 'left':
            # Tangent is 90 deg CCW from radial
            tangent_angle = math.atan2(dx, -dy)
        else:
            # Tangent is 90 deg CW from radial
            tangent_angle = math.atan2(-dx, dy)

        error = bike_heading - tangent_angle
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        return error

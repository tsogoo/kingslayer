import { Point } from './kinematics'

export interface Animation {
	start: Point;
	end: Point;
	duration: number;
};

export interface Animate {
	point: Point;
	delay: number;
};

export interface Config {
	type: string;
	val: number | string;
};
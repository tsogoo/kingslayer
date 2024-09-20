import { Point, ArmParams, ArmConfigurable, NoneKinematics } from '../common/kinematics'

export interface ScaraArmParams extends ArmParams {
	L2: number;
	position: Point;
	color: string;
	opposite?: boolean;
};

export class ScaraArm extends ArmConfigurable<ScaraArmParams> {
	constructor(conf: ScaraArmParams) {
		super(new ScaraKinematics(conf))
	}
	getType(): string {
		return 'Scara';
	}
};

export class ScaraKinematics extends NoneKinematics<ScaraArmParams> {

	constructor(conf: ScaraArmParams) {
		super(conf);
	}

	getPoints(point: Point): Point[] {
		const to = this.preProcessPoint(point);

		const p = this.calculateXY(to);
		let x = p.x;
		let y = p.y;
		const points: Point[] = [
			{
				x: 0, y: 0
			}, {
				x: x, y: y
			}, {
				x: to.x, y: to.y
			}
		];

		return this.postProcessPoint(points);
	};

	getAngle(point: Point): number {
		const to = this.preProcessPoint(point);
		let p = this.calculateXY(to);
		let angle: number = Math.atan2(p.y, p.x);
		// if (angle < 0)
		//     angle+=2*Math.PI;
		return angle;
	};

	preProcessPoint(point: Point): Point {
		let arm = this.getConf();
		return {
			x: point.x - arm.position.x,
			y: point.y - arm.position.y
		};
	}

	postProcessPoint(points: Point[]): Point[] {
		let arm = this.getConf();
		points.forEach((point) => { point.x += arm.position.x; point.y += arm.position.y; });
		return points;
	}

	calculateXY(to: Point): Point {
		const arm = this.getConf();
		const k: number = (to.x ** 2 + to.y ** 2 + arm.L1 ** 2 - arm.L2 ** 2) / 2;
		const y: number = (k * to.y + (arm.opposite ? -1 : 1) * to.x * Math.sqrt(arm.L1 ** 2 * (to.x ** 2 + to.y ** 2) - k ** 2)) / (to.x ** 2 + to.y ** 2);
		const x: number = (k - to.y * y) / to.x;
		return { x: x, y: y };
	}
};

export class ScaraKinematicsTrajectory {
	conf: ScaraArmParams[];
	constructor(conf: ScaraArmParams[]) {
		this.conf = conf;
	}
	calculate(angles: number[]): Point {
		const angle1 = angles[0];
		const angle2 = angles[1];
		const arms = this.conf;
		const arm1 = arms[0];
		const arm2 = arms[1];

		//  calculate x,y from given angle1, angle2
		const y1 = arm1.L1 * Math.sin(angle1);
		const x1 = arm1.L1 * Math.cos(angle1);
		const xarm2 = arm2.position.x - arm1.position.x;
		const yarm2 = arm2.position.y - arm1.position.y;
		const y2 = arm2.L1 * Math.sin(angle2) + yarm2;
		const x2 = arm2.L1 * Math.cos(angle2) + xarm2;
		const k = (arm1.L2 ** 2 - (x1 ** 2 + y1 ** 2) - (arm2.L2 ** 2 - (x2 ** 2 + y2 ** 2))) / 2;
		const dx = x2 - x1;
		const dy = y2 - y1;
		const a = 1 + (dx / dy) ** 2;
		const b = -2 * k * dx / (dy ** 2) + 2 * (x2 * y1 - x1 * y2) / dy;
		const c = x1 ** 2 + (k ** 2) / (dy ** 2) - 2 * y1 * k / dy + y1 ** 2 - arm1.L2 ** 2;
		let x = (-b + Math.sqrt(b ** 2 - 4 * a * c)) / (2 * a);
		let y = (k - x * dx) / dy;
		x += arm1.position.x;
		y += arm1.position.y;
		return {
			x: x, y: y
		}
	}

	convert(angle: number): number {
		return angle / Math.PI * 180;
	}
}
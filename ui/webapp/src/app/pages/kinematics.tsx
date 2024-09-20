"use client";

import React, { useEffect, useState } from 'react';
import { ArmParams, Point, x0y0 } from '../common/kinematics'
import { Animate, Animation } from '../common/ui'
import { KinematicsContext } from '../context/kinematics';
import { ArmComponent } from '../components/arm';
import { AnimationComponent } from '../components/animation';
import { ScaraKinematics, ScaraKinematicsTrajectory } from '../kinematics/scara';

interface KinematicsComponentConf {
	conf: KinematicsConf;
}

interface KinematicsConf {
	offset: Point;
	arms: ArmParams[];
	init?: Point;
	animation: Animation;
}

const KinematicsComponent: React.FC<KinematicsComponentConf> = ({ conf }) => {

	const [position, setPosition] = useState<Point>(x0y0);
	const [offset] = useState<Point>(conf.offset);
	const [animation, setAnimation] = useState<Animation>(conf.animation);

	const onSvgClick = (event: React.MouseEvent<SVGSVGElement, MouseEvent>) => {
		const svgElement = event.currentTarget;
		const rect = svgElement.getBoundingClientRect();
		const x = event.clientX - rect.left;
		const y = event.clientY - rect.top;
		setPosition({ x: x - conf.offset.x, y: y - conf.offset.y });
	};

	useEffect(() => {
		if (conf.init) {
			setPosition(conf.init);
		}
	}, []);

	const initAnimation = () => {
		animate(custom());
		// animate(linear());
	}

	const linear = (): Animate[] => {
		let iter = 1;
		let x = animation.start.x;
		let y = animation.start.y;
		let delta_x = (animation.end.x - animation.start.x) / iter;
		let delta_y = (animation.end.y - animation.start.y) / iter;
		let animations: Animate[] = []
		for (let i = 0; i < iter; i++) {
			animations.push({ point: { x: x, y: y }, delay: 30 });
			x += delta_x;
			y += delta_y;
		}
		return animations;
	}

	const custom = (): Animate[] => {
		let confs = [
			{
				type: 'Scara',
				L1: 100,
				L2: 100,
				position: { x: 0, y: 0 },
				color: 'green',
			},
			{
				type: 'Scara',
				L1: 100,
				L2: 100,
				position: { x: 20, y: 0 },
				color: 'red',
				opposite: true
			}
		];
		let arm1 = confs[0];
		let arm2 = confs[1];
		let kin1 = new ScaraKinematics(arm1);
		let kin2 = new ScaraKinematics(arm2);

		let start = animation.start;
		let end = animation.end;

		let angle_s1 = kin1.getAngle(start);
		let angle_s2 = kin2.getAngle(start);
		let angle_e1 = kin1.getAngle(end);
		let angle_e2 = kin2.getAngle(end);

		const iter: number = 100;
		let delta_1 = (angle_e1 - angle_s1) / iter;
		let delta_2 = (angle_e2 - angle_s2) / iter;

		let trajectory = new ScaraKinematicsTrajectory(confs);

		let animations: Animate[] = []
		let a1 = angle_s1;
		let a2 = angle_s2;
		let p: Point;

		for (let i = 0; i < iter; i++) {
			p = trajectory.calculate([a1, a2]);
			animations.push({ point: { x: p.x, y: p.y }, delay: 30 });
			a1 += delta_1;
			a2 += delta_2;
		}
		return animations;
	}

	const animate = (animations: Animate[]) => {
		const conf = animations.shift();
		if (conf) {
			setTimeout(function () {
				setPosition(conf.point);
				animate(animations);
			}, conf.delay);
		}
	}
	useEffect(() => {
		initAnimation();
	}, [animation]);
	return (
		<KinematicsContext.Provider value={{
			position, offset,
			setPosition,
			animation, setAnimation
		}}>
			<>
				<div style={{ float: 'left', border: '1px solid' }}>
					<svg onClick={onSvgClick} height="600" width="1000" xmlns="http://www.w3.org/2000/svg">
						{
							conf.arms.map((arm, i) => (
								<ArmComponent key={i} conf={arm} />
							))
						}
						{
							conf.animation && conf.animation.end &&
							<>
								<circle r="3" cx={conf.animation.start.x + offset.x} cy={conf.animation.start.y + offset.y} fill="brown" />
								<circle r="3" cx={conf.animation.end.x + offset.x} cy={conf.animation.end.y + offset.y} fill="brown" />
							</>
						}
					</svg>
				</div>
				<div style={{ float: 'right' }}>
					{
						conf.animation &&
						<div style={{ float: 'right' }}>
							<AnimationComponent conf={conf.animation} />
						</div>
					}
				</div>
			</>
		</KinematicsContext.Provider>
	);
};

export default KinematicsComponent;

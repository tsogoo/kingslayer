"use client";

import React, { useEffect, useState, useContext } from 'react';
import { KinematicsContext } from '../context/kinematics';
import { ScaraArm, ScaraArmParams } from '../kinematics/scara';
import { Point } from '../common/kinematics';

interface ScaraArmComponentConf {
	conf: ScaraArmParams;
}

export const ScaraArmComponent: React.FC<ScaraArmComponentConf> = ({ conf }) => {
	const [pathStr, setPathStr] = useState<string>('');
	const [points, setPoints] = useState<Point[]>([])
	const [angle, setAngle] = useState(0)
	const [arm, setArm] = useState<ScaraArm>(new ScaraArm(conf));
	const [trajectoryStr, setTrajectoryStr] = useState<string>('');
	const [trajectory, setTrajectory] = useState<Point[]>([]);

	const { position, offset } = useContext(KinematicsContext);

	const calculatePoints = () => {
		let points = arm.kinematics.getPoints(position);
		let angle = arm.kinematics.getAngle(position);
		setPoints(points);
		setAngle(angle);
	}
	useEffect(() => {
		calculatePoints();
		trajectory.push(position);
	}, [position, offset, arm]);
	useEffect(() => {
		setPathStr(points.map((point, i) => `${i == 0 ? 'M' : 'L'}${point.x + offset.x} ${point.y + offset.y}`).join(' '));
		setTrajectoryStr(trajectory.map((point, i) =>
			i<2 ? `` : `${i == 2 ? 'M' : 'L'}${point.x + offset.x} ${point.y + offset.y}`
).join(' '))
	}, [points]);
	return (
		<>
			<path d={pathStr} fill="none" stroke={arm.getConf().color} strokeWidth="2" />
			<text
				x={arm.getConf().position.x + offset.x - 20 + (arm.getConf().opposite ? 1 : -1) * 40}
				y={arm.getConf().position.y + offset.y - 20} fontSize="20">{Math.round(angle / Math.PI * 180 * 100) / 100}°</text>
			<path d={trajectoryStr} fill="none" stroke="black" strokeWidth="2" />
		</>
	);
};
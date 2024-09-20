"use client";

import React, { useEffect, useState, useContext } from 'react';
import { Animation, Animate, Config } from '../common/ui'
import { KinematicsContext } from '../context/kinematics';
import { AnimationContext } from '../context/animation';

interface AnimationConf {
	conf: Animation
}

export const AnimationComponent: React.FC<AnimationConf> = ({ conf }) => {
	const { animation, setAnimation } = useContext(KinematicsContext);

	const getAnimation = (): Animation => {
		return {
			start: {
				x: animation.start.x,
				y: animation.start.y
			},
			end: {
				x: animation.end.x,
				y: animation.end.y
			},
			duration: animation.duration
		}
	};

	const setValue = (type: string, val: any) => {
		let animation = getAnimation();
		switch (type) {
			case 'x':
				animation.start.x = val;
				break;
			case 'y':
				animation.start.y = val;
				break;
			case 'x1':
				animation.end.x = val;
				break;
			case 'y1':
				animation.end.y = val;
				break;
		}
		setAnimation(animation);
	};
	const initAnimation = () => {
		setAnimation(getAnimation());
	}
	useEffect(() => {
		initAnimation();
	}, []);
	return (
		<AnimationContext.Provider value={{ setValue }}>
			<div>Animation</div>
			<div>
				<div>from:</div>
				<ValueComponent conf={{ type: 'x', val: conf.start.x }} />
				<ValueComponent conf={{ type: 'y', val: conf.start.y }} />
				<div>to:</div>
				<ValueComponent conf={{ type: 'x1', val: conf.end.x }} />
				<ValueComponent conf={{ type: 'y1', val: conf.end.y }} />
			</div>
		</AnimationContext.Provider>
	);
}

interface ValueComponentConf {
	conf: Config;
}

const ValueComponent: React.FC<ValueComponentConf> = ({ conf }) => {
	const [v, setV] = useState<string | number>(conf.val);
	const [v2, setV2] = useState(v);
	const { setValue } = useContext(AnimationContext);

	const onChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const val = Number(event.target.value);
		setV(val);
	};

	useEffect(() => {
		const handler = setTimeout(() => {
			setV2(v);
		}, 500);
		return () => {
			clearTimeout(handler);
		};
	}, [v]);
	useEffect(() => {
		setValue(conf.type, v2);
	}, [v2]);
	return (
		<div>
			<span>{conf.type}</span>
			<input
				type="text"
				value={v}
				onChange={onChange}
			/>
		</div>
	);
}
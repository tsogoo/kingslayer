"use client";

import React, { useEffect, useState, useContext } from 'react';
import { Arm, Point, KinematicsContext, x0y0 } from '../context/kinematics';

interface ArmComponentConf {
    conf: ArmConf
}

interface ArmConf {
    arm: Arm;
}

const ArmComponent: React.FC<ArmComponentConf> = ({conf}) => {
    const [pathStr, setPathStr] = useState<string>('');
    const [points, setPoints] = useState<Point[]>([])
    const [angle, setAngle] = useState(0)

    const { position, offset } = useContext(KinematicsContext);

    const calculatePoints = () => {
        const arm = conf.arm
        //  relative to arm starting point
        const to: Point = {
            x:position.x-arm.position.x,
            y:position.y-arm.position.y
        };
        const k: number = (to.x**2+to.y**2+arm.L1**2-arm.L2**2)/2;
        const y: number = (k*to.y+(arm.opposite?-1:1)*to.x*Math.sqrt(arm.L1**2*(to.x**2+to.y**2)-k**2))/(to.x**2+to.y**2);
        const x: number = (k-to.y*y)/to.x;
        let angle: number = Math.round(Math.atan2(y,x)/Math.PI*180);
        if (angle < 0)
            angle+=360;
        const points: Point[] = [
            {
                x:0, y:0
            }, {
                x:x, y:y
            }, {
                x:to.x, y:to.y
            }
        ];
        points.forEach((point) => { point.x += arm.position.x; point.y += arm.position.y; })
        setPoints(points);
        setAngle(angle);
    }
    useEffect(() => {
        calculatePoints();
    }, [position, offset]);
    useEffect(() => {
        setPathStr(points.map((point,i)=>`${i==0?'M':'L'}${point.x+offset.x} ${point.y+offset.y}`).join(' '));
    }, [points]);
    return (
        <>
        <path d={pathStr} fill="none" stroke={conf.arm.color} strokeWidth="2" />
        <text
            x={conf.arm.position.x+offset.x-20}
            y={conf.arm.position.y+offset.y-20} fontSize="20">{angle}°</text>
        </>
    );
};

interface KinematicsComponentConf {
    conf: KinematicsConf;
}

interface KinematicsConf {
    offset: Point;
    arms: Arm[];
    init: Point;
}

const KinematicsComponent: React.FC<KinematicsComponentConf> = ({conf}) => {
    
    const [position, setPosition] = useState<Point>(x0y0);
    const [offset, setOffset] = useState<Point>(conf.offset)

    const onSvgClick = (event: React.MouseEvent<SVGSVGElement, MouseEvent>) => {
        const svgElement = event.currentTarget;
        const rect = svgElement.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        setPosition({x:x-conf.offset.x,y:y-conf.offset.y});
    };
    useEffect(() => {
        setPosition(conf.init)
    }, [])
    return (
        <KinematicsContext.Provider value={{position, offset}}>
            <div style={{border:'1px solid'}}>
                <svg onClick={onSvgClick} height="1200" width="1200" xmlns="http://www.w3.org/2000/svg">
                {
                    conf.arms.map((arm, i) => (
                        <ArmComponent key={i} conf={{arm:arm}}/>
                    ))
                }
                </svg>
            </div>
        </KinematicsContext.Provider>
    );
};

export default KinematicsComponent;

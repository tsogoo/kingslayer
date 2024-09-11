"use client";

import React, { useEffect, useState, useContext } from 'react';
import { Arm, Point, KinematicsContext, x0y0 } from '../context/kinematics';

interface ArmComponentConf {
    conf: Arm
}

const ArmComponent: React.FC<ArmComponentConf> = ({conf}) => {
    const [pathStr, setPathStr] = useState<string>('');
    const [points, setPoints] = useState<Point[]>([])

    const { position, offset } = useContext(KinematicsContext);

    const calculatePoints = () => {
        const arm = conf
        //  relative to arm starting point
        const to: Point = {
            x:position.x-conf.position.x,
            y:position.y-conf.position.y
        };
        const k: number = (to.x**2+to.y**2+arm.L1**2-arm.L2**2)/2;
        const y: number = (k*to.y+(conf.opposite?-1:1)*to.x*Math.sqrt(arm.L1**2*(to.x**2+to.y**2)-k**2))/(to.x**2+to.y**2);
        const x: number = (k-to.y*y)/to.x;
        const points: Point[] = [
            {
                x:0, y:0
            }, {
                x:x, y:y
            }, {
                x:to.x, y:to.y
            }
        ];
        points.forEach((point) => { point.x += conf.position.x; point.y += conf.position.y; })
        setPoints(points);
    }
    useEffect(() => {
        calculatePoints();
    }, [position, offset]);
    useEffect(() => {
        setPathStr(points.map((point,i)=>`${i==0?'M':'L'}${point.x+offset.x} ${point.y+offset.y}`).join(' '));
    }, [points]);
    return (
        <>
        <path d={pathStr} fill="none" stroke={conf.color} strokeWidth="2" />
        {
            points.map((point, i) => (
                <text key={i} x={point.x+offset.x} y={point.y+offset.y} font-size="24">P{i}</text>
            ))
        }
        </>
    );
};

interface KinematicsComponentConf {
    conf: KinematicsConf;
}

interface KinematicsConf {
    offset: Point;
    arms: Arm[];
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
    return (
        <KinematicsContext.Provider value={{position, offset}}>
            <div style={{border:'1px solid'}}>
                <svg onClick={onSvgClick} height="1200" width="1200" xmlns="http://www.w3.org/2000/svg">
                {
                    conf.arms.map((arm, i) => (
                        <ArmComponent key={i} conf={arm}/>
                    ))
                }
                </svg>
            </div>
        </KinematicsContext.Provider>
    );
};

export default KinematicsComponent;
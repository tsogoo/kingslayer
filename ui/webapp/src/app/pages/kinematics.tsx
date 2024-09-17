"use client";

import React, { useEffect, useState, useContext } from 'react';
import { Arm, Point, KinematicsContext, x0y0, Animation, Animate, Config, AnimationContext } from '../context/kinematics';

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

interface AnimationConf {
    conf: Animation
}

const AnimationComponent: React.FC<AnimationConf> = ({conf}) => {
    const [ animation, setAnimation ] = useState<Animation>(conf)
    const { setPosition } = useContext(KinematicsContext);

    const getAmination = (): Animation => {
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
    
    const setConfig = (type: string, val: any) => {
        let animation = getAmination();
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
        const vx = (animation.end.x - animation.start.x)/animation.duration;
        const vy = (animation.end.y - animation.start.y)/animation.duration;
        let x, y = 0;
        let animations: Animate[] = []
        let delay: number = animation.duration/50;
        for (let time = 0; time <= animation.duration; time+=delay) {
            x = animation.end.x-vx*time;
            y = animation.end.y-vy*time;
            animations.push({point:{x:x, y:y},delay:delay});
        }
        animate(animations);
    }
    const animate = (animations: Animate[]) => {
        const conf = animations.pop();
        if (conf) {
            setTimeout(function() {
                setPosition(conf.point);
                animate(animations);
            }, conf.delay);
        }
    }
    useEffect(() => {
        initAnimation();
    }, []);
    useEffect(() => {
        initAnimation();
    }, [animation]);
    return (
        <AnimationContext.Provider value={{setConfig}}>
            <div>Animation</div>
            <div>
                <ConfigComponent conf={{type:'x',val:conf.start.x}} />
                <ConfigComponent conf={{type:'y',val:conf.start.y}} />
                <ConfigComponent conf={{type:'x1',val:conf.end.x}} />
                <ConfigComponent conf={{type:'y1',val:conf.end.y}} />
            </div>
        </AnimationContext.Provider>
    );
}

interface ConfigComponentConf {
    conf: Config;
}

const ConfigComponent: React.FC<ConfigComponentConf> = ({conf}) => {
    const [v, setV] = useState<string|number>(conf.val);
    const [v2, setV2] = useState(v);
    const { setConfig } = useContext(AnimationContext);
    
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
        setConfig(conf.type, v2);
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

interface KinematicsComponentConf {
    conf: KinematicsConf;
}

interface KinematicsConf {
    offset: Point;
    arms: Arm[];
    init?: Point;
    animation?: Animation;
}

const KinematicsComponent: React.FC<KinematicsComponentConf> = ({conf}) => {
    
    const [position, setPosition] = useState<Point>(x0y0);
    const [offset, setOffset] = useState<Point>(conf.offset);
    
    const onSvgClick = (event: React.MouseEvent<SVGSVGElement, MouseEvent>) => {
        const svgElement = event.currentTarget;
        const rect = svgElement.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        setPosition({x:x-conf.offset.x,y:y-conf.offset.y});
    };
    
    useEffect(() => {
        if (conf.init) {
            setPosition(conf.init);
        }
    }, []);
    return (
        <KinematicsContext.Provider value={{position, offset, setPosition}}>
            <>
                <div style={{float:'left',border:'1px solid'}}>
                    <svg onClick={onSvgClick} height="600" width="1000" xmlns="http://www.w3.org/2000/svg">
                    {
                        conf.arms.map((arm, i) => (
                            <ArmComponent key={i} conf={{arm:arm}}/>
                        ))
                    }
                    </svg>
                </div>
                {
                    conf.animation &&
                    <div style={{float:'right'}}>
                        <AnimationComponent conf={conf.animation}/>
                    </div>
                }
            </>
        </KinematicsContext.Provider>
    );
};

export default KinematicsComponent;

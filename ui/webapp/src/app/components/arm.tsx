"use client";

import React, { useEffect, useState, useContext } from 'react';
import { ScaraArmComponent } from './scara';

interface ArmComponentConf {
	conf: any
}

export const ArmComponent: React.FC<ArmComponentConf> = ({ conf }) => {
	return (
		<>
			{
				conf.type == 'Scara' &&
				<ScaraArmComponent conf={conf} />
			}
		</>
	);
};
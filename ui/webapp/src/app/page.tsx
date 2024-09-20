// import Board from './board'
import { ScaraArm } from './kinematics/scara';
import KinematicsComponent from './pages/kinematics';

export default function Home() {
  const kinematicsConf = {
    offset: {
      x: 400, y: 300
    },
    arms: [
      {
        type: 'Scara',
        L1:100,
        L2:100,
        position: {x:0, y:0},
        color: 'green',
      },
      {
        type: 'Scara',
        L1:100,
        L2:100,
        position: {x:20,y:0},
        color: 'red',
        opposite: true
      }
    ],
    animation: {
      start: {
        x:50, y:50
      },
      end: {
        x:100, y:80
      },
      duration: 5000
    }
  }

  return (
    <main>
      {/* <Board/> */}
      <KinematicsComponent conf={kinematicsConf}/>
    </main>
  );
}

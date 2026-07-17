module Saver where

import Control.Monad.Reader
import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Binary
import Data.ByteString
import Misc
import ML.ANN.Block
import ML.ANN.BString
import ML.ANN.LayerBS
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import Types

runSaver :: Int -> Mon ()
runSaver i = do
    let ofile = "/tmp/results/ann-" P.++ (show i) P.++ ".ann"
    block <- gets stBlock
    blinfo <- gets stBLInfo
    phase <- gets stPhase
    to <- gets stFilesToOpen
    if phase P.== Save
    then do
        modify (\s -> s { stFilesToOpen = ofile : to , stPhase = Save1})
    else if phase P.== Save1
    then do
        modify (\s -> s { stPhase = Done})
        writer ofile (toStrict (encode (block2network blinfo (use block))))
        closer ofile
    else 
        return ()



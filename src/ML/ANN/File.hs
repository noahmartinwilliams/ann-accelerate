module ML.ANN.File(block2bs) where

import ML.ANN.Block
import ML.ANN.Mat
import ML.ANN.Vect

import Data.Array.Accelerate as A
import Data.Array.Accelerate.IO.Data.ByteString as A
import Prelude as P

import Data.ByteString as B
import Data.ByteString.Builder as B
import Data.ByteString.Conversion as B
import Data.ByteString.Lazy as BL
import Data.Serialize

block2bs :: (BlockInfo, BlockV) -> BL.ByteString
block2bs (blinfo, (vecti, vectd)) = do
    let ver = BL.pack [0, 1, 0]
        blinfobs = fromStrict (encode blinfo)
        vectibs = fromStrict (A.toByteStrings vecti)
        vectdbs = fromStrict (A.toByteStrings vectd)
    ver <> blinfobs <> vectibs <> vectdbs 

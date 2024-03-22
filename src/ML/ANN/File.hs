module ML.ANN.File(block2bs, bs2block) where

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
    let ver = BL.pack [127, 65, 78, 78, 0, 0, 1, 0]
        vectil = A.toList vecti
        vectdl = A.toList vectd
    ver <> (fromStrict (encode (blinfo, vectil, vectdl)))

bs2block :: MonadFail m => BL.ByteString -> m (BlockInfo, BlockV)
bs2block bs = do
    let (_: 65: 78: 78: 0: major: minor: bug:_) = BL.unpack bs
    (blinfo, vecti, vectd) <- getInfo major minor bug (BL.drop 8 bs)
    return (blinfo, (vecti, vectd))

getInfo :: MonadFail m => Word8 -> Word8 -> Word8 -> BL.ByteString -> m (BlockInfo, Vector Int, Vector Double)
getInfo 0 1 0 bs = do
    (blinfo, vectil, vectdl) <- intern (decode (toStrict bs))
    let vecti = A.fromList (Z:.(P.length vectil)) vectil
        vectd = A.fromList (Z:.(P.length vectdl)) vectdl
    return (blinfo, vecti, vectd) where
        intern :: MonadFail m => Either String (BlockInfo, [Int], [Double]) -> m (BlockInfo, [Int], [Double])
        intern (Left str) = fail str
        intern (Right a) = return a
getInfo major m b _ = error ("Version " P.++ (show major) P.++ "." P.++ (show m) P.++ "." P.++ (show b) P.++ " is not supported.")

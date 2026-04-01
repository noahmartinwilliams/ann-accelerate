module Samps where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.LLVM.PTX as PTX
import Data.List.Split
import qualified Data.ByteString as B
import Prelude as P

bs2Answer :: B.ByteString -> Matrix Double
bs2Answer bs | (B.head bs) P.== 0 = fromList (Z:.10:.1) [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 1 = fromList (Z:.10:.1) [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 2 = fromList (Z:.10:.1) [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 3 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 4 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 5 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 6 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 7 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 8 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
bs2Answer bs | (B.head bs) P.== 9 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

combineAnswers :: [Matrix Double] -> Matrix Double
combineAnswers (h : ls) = I.run (P.foldl (A.++) (use h) (P.map use ls))

mkSamps :: Int -> B.ByteString -> B.ByteString -> [(Matrix Double, Matrix Double)]
mkSamps mbs imgs answers = do
    let splitImgs = bsSplitEvery (mbs * 28 * 28) imgs
        imgMats = P.map (bs2Mat mbs) splitImgs
        answerMats = bsSplitEvery mbs answers
        answerMats' = chunksOf mbs (P.map bs2Answer answerMats)
        answerMats'' = P.map combineAnswers answerMats'
    P.zip imgMats answerMats''

bsSplitEvery :: Int -> B.ByteString -> [B.ByteString]
bsSplitEvery i bs | (B.null bs) = []
bsSplitEvery i bs = (B.take i bs ) : (bsSplitEvery i (B.drop i bs))

bs2Mat :: Int -> B.ByteString -> Matrix Double
bs2Mat mbs bs = do
    let uped = B.unpack bs
        asDoubles = P.map (\x -> P.fromIntegral x :: Double) uped
        scaled = P.map (\x -> (x - 128.0) / 128.0) asDoubles
    A.fromList (Z:.(28*28):.mbs) scaled

module Samps where

import Control.Parallel.Strategies
import Data.Array.Accelerate as A
import qualified Data.ByteString as B
import Data.List.Split
import Types
import GHC.Conc
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
combineAnswers ms = do
    let ls = P.foldr (P.++) [] (P.map (A.toList) ms)
        l = P.length ms
    A.fromList (Z:.1:.10*l) ls

getSamps :: Int -> B.ByteString -> B.ByteString -> [(Matrix Double, Matrix Double)]
getSamps mbs imgs answers = do
    let splitImgs = bsSplitEvery (mbs * 28 * 28) imgs
        imgMats = P.map (bs2Mat mbs) splitImgs
        answerMats = bsSplitEvery mbs answers
        answerMats' = chunksOf mbs (P.map bs2Answer answerMats)
        answerMats'' = P.map combineAnswers answerMats'
    (P.zip imgMats answerMats'') `using` (parListChunk numCapabilities rdeepseq)

bsSplitEvery :: Int -> B.ByteString -> [B.ByteString]
bsSplitEvery i bs | (B.null bs) = []
bsSplitEvery i bs = (B.take i bs ) : (bsSplitEvery i (B.drop i bs))

bs2Mat :: Int -> B.ByteString -> Matrix Double
bs2Mat mbs bs = do
    let uped = B.unpack bs
        asDoubles = P.map (\x -> P.fromIntegral x :: Double) uped
        scaled = P.map (\x -> (x / 255.0) ) asDoubles
    (A.fromList (Z:.mbs:.(28*28)) scaled )

prepareSamp :: Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double)
prepareSamp m = do
    let (l1, l2) = A.unlift m :: (Acc (Matrix Double), Acc (Matrix Double))
    A.lift (A.transpose l1, A.transpose l2)

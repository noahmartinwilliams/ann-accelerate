{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}
module Conf where

import Control.Monad.Reader
import Data.Aeson 
import Data.Array.Accelerate as A
import qualified Data.ByteString as B
import Data.List.Split
import Data.Maybe
import GHC.Generics
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.IO
import System.Random

data Conf = Conf { inputAF :: String, layers :: String, optimizer :: String, lr :: Double, beta1 :: Double, beta2 :: Double, costF :: String, numEpochs :: Int, miniBatchSize :: Int } deriving(Generic, Show)

instance ToJSON Conf where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Conf 

getConf :: String -> Maybe Conf
getConf inp = do
    let bs = fromString inp
    Data.Aeson.decode bs :: Maybe Conf

type Fn = ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Vector Double, Vector Int, Vector Double))

module Misc where

import Control.Monad.State
import Data.ByteString
import Data.Map
import Types

writer :: String -> ByteString -> Mon ()
writer str input = do
    m <- gets stFilesToWrite
    let res = Data.Map.insert str input m
    modify (\s -> s { stFilesToWrite = res})

closer :: String -> Mon ()
closer str = do
    closers <- gets stFilesToClose
    modify (\s -> s { stFilesToClose = str : closers})

module SideEff where

import Data.ByteString as BS
import Data.Map
import Prelude as P
import System.IO
import System.Mem
import Types

doIO :: St -> IO St
doIO st = do
    st' <- openFiles st
    st'' <- writeFiles st'
    st2 <- closeFiles st''
    performMajorGC
    return st2

openFiles :: St -> IO St
openFiles s@(St { stFilesToOpen = []}) = return s
openFiles s@(St { stFilesToOpen = (sfto : r)}) = do
    h <- openFile sfto WriteMode
    let m = stFiles s
    let m' = insert sfto h m
    hSetBuffering h LineBuffering
    openFiles (s { stFilesToOpen = r, stFiles = m'})

writeFiles :: St -> IO St
writeFiles s@(St { stFilesToWrite = m }) | Data.Map.null m = return s
writeFiles s@(St { stFiles = fs, stFilesToWrite = m }) = do
    let l = Data.Map.toList m
        hs = P.map (\x -> (Data.Map.lookup (fst x) fs, snd x)) l
    writeFiles' hs 
    return (s { stFilesToWrite = Data.Map.empty }) where
        writeFiles' :: [(Maybe Handle, ByteString)] -> IO ()
        writeFiles' [] = return ()
        writeFiles' ((Nothing, _) : r) = writeFiles' r
        writeFiles' ((Just h, a) : r) = do
            BS.hPut h a
            writeFiles' r

closeFiles :: St -> IO St
closeFiles s@(St { stFilesToClose = []}) = return s
closeFiles s@(St { stFiles = m, stFilesToClose = (h : r) }) = do
    let hn = Data.Map.lookup h m
        m' = Data.Map.delete h m
    case hn of
        (Just hn') -> do
            hClose hn'
            closeFiles (s { stFilesToClose = r, stFiles = m'})
        Nothing ->
            closeFiles (s { stFilesToClose = r, stFiles = m'})

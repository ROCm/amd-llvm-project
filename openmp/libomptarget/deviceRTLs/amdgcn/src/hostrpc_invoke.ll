; ModuleID = 'hostrpc_invoke.ll'
source_filename = "llvm-link"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"
target triple = "amdgcn-amd-amdhsa"

%struct.__ockl_hostrpc_result_t = type { i64, i64, i64, i64, i64, i64, i64, i64 }
%struct.buffer_t = type { %struct.header_t addrspace(1)*, %struct.payload_t addrspace(1)*, %struct.hsa_signal_s, i64, i64, i32 }
%struct.header_t = type { i64, i64, i32, i32 }
%struct.payload_t = type { [64 x [8 x i64]] }
%struct.hsa_signal_s = type { i64 }

; Function Attrs: convergent norecurse nounwind
define %struct.__ockl_hostrpc_result_t @hostrpc_invoke(i32 %service_id, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %arg5, i64 %arg6, i64 %arg7) local_unnamed_addr #0 {
entry:
  tail call void asm sideeffect "; hostcall_invoke: record need for hostcall support\0A\09.type needs_hostcall_buffer,@object\0A\09.global needs_hostcall_buffer\0A\09.comm needs_hostcall_buffer,4", ""() #6, !srcloc !4
  %0 = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %arrayidx = getelementptr inbounds i8, i8 addrspace(4)* %0, i64 24
  %1 = bitcast i8 addrspace(4)* %arrayidx to %struct.buffer_t addrspace(1)* addrspace(4)*
  %2 = load %struct.buffer_t addrspace(1)*, %struct.buffer_t addrspace(1)* addrspace(4)* %1, align 8, !tbaa !5
  %call.i = tail call i32 @__ockl_lane_u32() #6
  %3 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %call.i) #4
  %cmp.i = icmp eq i32 %call.i, %3
  br i1 %cmp.i, label %if.then.i, label %pop_free_stack.exit

if.then.i:                                        ; preds = %entry
  %free_stack.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 3
  %4 = load atomic i64, i64 addrspace(1)* %free_stack.i syncscope("one-as") acquire, align 8
  %headers.i.i.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 0
  %5 = load %struct.header_t addrspace(1)*, %struct.header_t addrspace(1)* addrspace(1)* %headers.i.i.i, align 8, !tbaa !9
  %index_size.i.i.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 5
  %6 = load i32, i32 addrspace(1)* %index_size.i.i.i, align 8, !tbaa !14
  %7 = and i32 %6, 63
  %shl.mask.i.i.i.i80 = zext i32 %7 to i64
  %notmask.i.i.i.i81 = shl nsw i64 -1, %shl.mask.i.i.i.i80
  %sub.i.i.i.i82 = xor i64 %notmask.i.i.i.i81, -1
  %and.i.i.i.i83 = and i64 %4, %sub.i.i.i.i82
  %next.i.i84 = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %5, i64 %and.i.i.i.i83, i32 0
  %8 = load atomic i64, i64 addrspace(1)* %next.i.i84 syncscope("one-as") monotonic, align 8
  %9 = cmpxchg i64 addrspace(1)* %free_stack.i, i64 %4, i64 %8 syncscope("one-as") acquire monotonic
  %10 = extractvalue { i64, i1 } %9, 1
  br i1 %10, label %pop_free_stack.exit, label %if.end.i.i

if.end.i.i:                                       ; preds = %if.end.i.i, %if.then.i
  %11 = phi { i64, i1 } [ %17, %if.end.i.i ], [ %9, %if.then.i ]
  %12 = extractvalue { i64, i1 } %11, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1) #4
  %13 = load %struct.header_t addrspace(1)*, %struct.header_t addrspace(1)* addrspace(1)* %headers.i.i.i, align 8, !tbaa !9
  %14 = load i32, i32 addrspace(1)* %index_size.i.i.i, align 8, !tbaa !14
  %15 = and i32 %14, 63
  %shl.mask.i.i.i.i = zext i32 %15 to i64
  %notmask.i.i.i.i = shl nsw i64 -1, %shl.mask.i.i.i.i
  %sub.i.i.i.i = xor i64 %notmask.i.i.i.i, -1
  %and.i.i.i.i = and i64 %12, %sub.i.i.i.i
  %next.i.i = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %13, i64 %and.i.i.i.i, i32 0
  %16 = load atomic i64, i64 addrspace(1)* %next.i.i syncscope("one-as") monotonic, align 8
  %17 = cmpxchg i64 addrspace(1)* %free_stack.i, i64 %12, i64 %16 syncscope("one-as") acquire monotonic
  %18 = extractvalue { i64, i1 } %17, 1
  br i1 %18, label %pop_free_stack.exit.loopexit, label %if.end.i.i

pop_free_stack.exit.loopexit:                     ; preds = %if.end.i.i
  %19 = extractvalue { i64, i1 } %11, 0
  br label %pop_free_stack.exit

pop_free_stack.exit:                              ; preds = %pop_free_stack.exit.loopexit, %if.then.i, %entry
  %packet_ptr.0.i = phi i64 [ 0, %entry ], [ %4, %if.then.i ], [ %19, %pop_free_stack.exit.loopexit ]
  %conv.i = trunc i64 %packet_ptr.0.i to i32
  %shr.i = lshr i64 %packet_ptr.0.i, 32
  %conv2.i = trunc i64 %shr.i to i32
  %20 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %conv.i) #4
  %21 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %conv2.i) #4
  %conv3.i = zext i32 %21 to i64
  %shl.i = shl nuw i64 %conv3.i, 32
  %conv4.i = zext i32 %20 to i64
  %or.i = or i64 %shl.i, %conv4.i
  %headers.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 0
  %22 = load %struct.header_t addrspace(1)*, %struct.header_t addrspace(1)* addrspace(1)* %headers.i, align 8, !tbaa !9
  %index_size.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 5
  %23 = load i32, i32 addrspace(1)* %index_size.i, align 8, !tbaa !14
  %24 = and i32 %23, 63
  %shl.mask.i.i = zext i32 %24 to i64
  %notmask.i.i = shl nsw i64 -1, %shl.mask.i.i
  %sub.i.i = xor i64 %notmask.i.i, -1
  %and.i.i = and i64 %or.i, %sub.i.i
  %payloads.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 1
  %25 = load %struct.payload_t addrspace(1)*, %struct.payload_t addrspace(1)* addrspace(1)* %payloads.i, align 8, !tbaa !15
  %call.i59 = tail call i32 @__ockl_lane_u32() #6
  %26 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %call.i59) #4
  %27 = tail call i64 @llvm.read_register.i64(metadata !16) #6
  %cmp.i60 = icmp eq i32 %call.i59, %26
  br i1 %cmp.i60, label %if.then.i61, label %fill_packet.exit

if.then.i61:                                      ; preds = %pop_free_stack.exit
  %control2.i = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %22, i64 %and.i.i, i32 3
  %activemask.i = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %22, i64 %and.i.i, i32 1
  %service.i = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %22, i64 %and.i.i, i32 2
  store i32 %service_id, i32 addrspace(1)* %service.i, align 8, !tbaa !17
  store i64 %27, i64 addrspace(1)* %activemask.i, align 8, !tbaa !19
  store i32 1, i32 addrspace(1)* %control2.i, align 4, !tbaa !20
  br label %fill_packet.exit

fill_packet.exit:                                 ; preds = %if.then.i61, %pop_free_stack.exit
  %idxprom.i = zext i32 %call.i59 to i64
  %arraydecay.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 0
  store i64 %arg0, i64 addrspace(1)* %arraydecay.i, align 8, !tbaa !5
  %arrayidx4.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 1
  store i64 %arg1, i64 addrspace(1)* %arrayidx4.i, align 8, !tbaa !5
  %arrayidx5.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 2
  store i64 %arg2, i64 addrspace(1)* %arrayidx5.i, align 8, !tbaa !5
  %arrayidx6.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 3
  store i64 %arg3, i64 addrspace(1)* %arrayidx6.i, align 8, !tbaa !5
  %arrayidx7.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 4
  store i64 %arg4, i64 addrspace(1)* %arrayidx7.i, align 8, !tbaa !5
  %arrayidx8.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 5
  store i64 %arg5, i64 addrspace(1)* %arrayidx8.i, align 8, !tbaa !5
  %arrayidx9.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 6
  store i64 %arg6, i64 addrspace(1)* %arrayidx9.i, align 8, !tbaa !5
  %arrayidx10.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idxprom.i, i64 7
  store i64 %arg7, i64 addrspace(1)* %arrayidx10.i, align 8, !tbaa !5
  %call.i63 = tail call i32 @__ockl_lane_u32() #6
  %28 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %call.i63) #4
  %cmp.i64 = icmp eq i32 %call.i63, %28
  br i1 %cmp.i64, label %if.then.i72, label %push_ready_stack.exit

if.then.i72:                                      ; preds = %fill_packet.exit
  %ready_stack.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 4
  %29 = load atomic i64, i64 addrspace(1)* %ready_stack.i syncscope("one-as") monotonic, align 8
  %30 = load %struct.header_t addrspace(1)*, %struct.header_t addrspace(1)* addrspace(1)* %headers.i, align 8, !tbaa !9
  %31 = load i32, i32 addrspace(1)* %index_size.i, align 8, !tbaa !14
  %32 = and i32 %31, 63
  %shl.mask.i.i.i.i67 = zext i32 %32 to i64
  %notmask.i.i.i.i68 = shl nsw i64 -1, %shl.mask.i.i.i.i67
  %sub.i.i.i.i69 = xor i64 %notmask.i.i.i.i68, -1
  %and.i.i.i.i70 = and i64 %or.i, %sub.i.i.i.i69
  %next.i.i74 = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %30, i64 %and.i.i.i.i70, i32 0
  store i64 %29, i64 addrspace(1)* %next.i.i74, align 8, !tbaa !21
  %33 = cmpxchg i64 addrspace(1)* %ready_stack.i, i64 %29, i64 %or.i syncscope("one-as") release monotonic
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %push.exit.i78, label %if.end.i.i77

if.end.i.i77:                                     ; preds = %if.end.i.i77, %if.then.i72
  %35 = phi { i64, i1 } [ %37, %if.end.i.i77 ], [ %33, %if.then.i72 ]
  %36 = extractvalue { i64, i1 } %35, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1) #4
  store i64 %36, i64 addrspace(1)* %next.i.i74, align 8, !tbaa !21
  %37 = cmpxchg i64 addrspace(1)* %ready_stack.i, i64 %36, i64 %or.i syncscope("one-as") release monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %push.exit.i78, label %if.end.i.i77

push.exit.i78:                                    ; preds = %if.end.i.i77, %if.then.i72
  %coerce.dive.i = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 2, i32 0
  %39 = load i64, i64 addrspace(1)* %coerce.dive.i, align 8
  tail call void @__ockl_hsa_signal_add(i64 %39, i64 1, i32 3) #6
  br label %push_ready_stack.exit

push_ready_stack.exit:                            ; preds = %push.exit.i78, %fill_packet.exit
  %call.i56 = tail call i32 @__ockl_lane_u32() #6
  %40 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %call.i56) #4
  %cmp.i57 = icmp eq i32 %call.i56, %40
  %control1.i = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %22, i64 %and.i.i, i32 3
  br label %while.cond.i

while.cond.i:                                     ; preds = %if.end5.i, %push_ready_stack.exit
  br i1 %cmp.i57, label %if.then.i58, label %if.end.i

if.then.i58:                                      ; preds = %while.cond.i
  %41 = load atomic i32, i32 addrspace(1)* %control1.i syncscope("one-as") acquire, align 4
  %and.i.i.i = and i32 %41, 1
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i58, %while.cond.i
  %ready_flag.0.i = phi i32 [ %and.i.i.i, %if.then.i58 ], [ 1, %while.cond.i ]
  %42 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %ready_flag.0.i) #4
  %cmp3.i = icmp eq i32 %42, 0
  br i1 %cmp3.i, label %get_return_value.exit, label %if.end5.i

if.end5.i:                                        ; preds = %if.end.i
  tail call void @llvm.amdgcn.s.sleep(i32 1) #4
  br label %while.cond.i

get_return_value.exit:                            ; preds = %if.end.i
  %idx.ext.i = zext i32 %call.i56 to i64
  %43 = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 0
  %incdec.ptr.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 1
  %44 = load i64, i64 addrspace(1)* %43, align 8, !tbaa !5
  %incdec.ptr6.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 2
  %45 = load i64, i64 addrspace(1)* %incdec.ptr.i, align 8, !tbaa !5
  %incdec.ptr7.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 3
  %46 = load i64, i64 addrspace(1)* %incdec.ptr6.i, align 8, !tbaa !5
  %incdec.ptr8.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 4
  %47 = load i64, i64 addrspace(1)* %incdec.ptr7.i, align 8, !tbaa !5
  %incdec.ptr9.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 5
  %48 = load i64, i64 addrspace(1)* %incdec.ptr8.i, align 8, !tbaa !5
  %incdec.ptr10.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 6
  %49 = load i64, i64 addrspace(1)* %incdec.ptr9.i, align 8, !tbaa !5
  %incdec.ptr11.i = getelementptr inbounds %struct.payload_t, %struct.payload_t addrspace(1)* %25, i64 %and.i.i, i32 0, i64 %idx.ext.i, i64 7
  %50 = load i64, i64 addrspace(1)* %incdec.ptr10.i, align 8, !tbaa !5
  %51 = load i64, i64 addrspace(1)* %incdec.ptr11.i, align 8, !tbaa !5
  %call.i32 = tail call i32 @__ockl_lane_u32() #6
  %52 = tail call i32 @llvm.amdgcn.readfirstlane(i32 %call.i32) #4
  %cmp.i33 = icmp eq i32 %call.i32, %52
  br i1 %cmp.i33, label %if.then.i44, label %return_free_packet.exit

if.then.i44:                                      ; preds = %get_return_value.exit
  %53 = load i32, i32 addrspace(1)* %index_size.i, align 8, !tbaa !14
  %54 = and i32 %53, 63
  %shl.mask.i.i35 = zext i32 %54 to i64
  %shl.i.i = shl nuw i64 1, %shl.mask.i.i35
  %add.i.i = add i64 %shl.i.i, %or.i
  %cmp.i.i = icmp eq i64 %add.i.i, 0
  %cond.i.i = select i1 %cmp.i.i, i64 %shl.i.i, i64 %add.i.i
  %free_stack.i36 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t addrspace(1)* %2, i64 0, i32 3
  %55 = load atomic i64, i64 addrspace(1)* %free_stack.i36 syncscope("one-as") monotonic, align 8
  %56 = load %struct.header_t addrspace(1)*, %struct.header_t addrspace(1)* addrspace(1)* %headers.i, align 8, !tbaa !9
  %notmask.i.i.i.i40 = shl nsw i64 -1, %shl.mask.i.i35
  %sub.i.i.i.i41 = xor i64 %notmask.i.i.i.i40, -1
  %and.i.i.i.i42 = and i64 %cond.i.i, %sub.i.i.i.i41
  %next.i.i46 = getelementptr inbounds %struct.header_t, %struct.header_t addrspace(1)* %56, i64 %and.i.i.i.i42, i32 0
  store i64 %55, i64 addrspace(1)* %next.i.i46, align 8, !tbaa !21
  %57 = cmpxchg i64 addrspace(1)* %free_stack.i36, i64 %55, i64 %cond.i.i syncscope("one-as") release monotonic
  %58 = extractvalue { i64, i1 } %57, 1
  br i1 %58, label %return_free_packet.exit, label %if.end.i.i49

if.end.i.i49:                                     ; preds = %if.end.i.i49, %if.then.i44
  %59 = phi { i64, i1 } [ %61, %if.end.i.i49 ], [ %57, %if.then.i44 ]
  %60 = extractvalue { i64, i1 } %59, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1) #4
  store i64 %60, i64 addrspace(1)* %next.i.i46, align 8, !tbaa !21
  %61 = cmpxchg i64 addrspace(1)* %free_stack.i36, i64 %60, i64 %cond.i.i syncscope("one-as") release monotonic
  %62 = extractvalue { i64, i1 } %61, 1
  br i1 %62, label %return_free_packet.exit, label %if.end.i.i49

return_free_packet.exit:                          ; preds = %if.end.i.i49, %if.then.i44, %get_return_value.exit
  %oldret = insertvalue %struct.__ockl_hostrpc_result_t undef, i64 %44, 0
  %oldret19 = insertvalue %struct.__ockl_hostrpc_result_t %oldret, i64 %45, 1
  %oldret21 = insertvalue %struct.__ockl_hostrpc_result_t %oldret19, i64 %46, 2
  %oldret23 = insertvalue %struct.__ockl_hostrpc_result_t %oldret21, i64 %47, 3
  %oldret25 = insertvalue %struct.__ockl_hostrpc_result_t %oldret23, i64 %48, 4
  %oldret27 = insertvalue %struct.__ockl_hostrpc_result_t %oldret25, i64 %49, 5
  %oldret29 = insertvalue %struct.__ockl_hostrpc_result_t %oldret27, i64 %50, 6
  %oldret31 = insertvalue %struct.__ockl_hostrpc_result_t %oldret29, i64 %51, 7
  ret %struct.__ockl_hostrpc_result_t %oldret31
}

; Function Attrs: nounwind readnone speculatable
declare align 4 i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: convergent
declare i32 @__ockl_lane_u32() local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.amdgcn.readfirstlane(i32) #3

; Function Attrs: nounwind
declare void @llvm.amdgcn.s.sleep(i32 immarg) #4

; Function Attrs: nounwind readonly
declare i64 @llvm.read_register.i64(metadata) #5

; Function Attrs: convergent
declare void @__ockl_hsa_signal_add(i64, i64, i32) local_unnamed_addr #2

attributes #0 = { convergent norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind readnone }
attributes #4 = { nounwind }
attributes #5 = { nounwind readonly }
attributes #6 = { convergent nounwind }

!opencl.ocl.version = !{!0, !0}
!llvm.ident = !{!1, !1}
!llvm.module.flags = !{!2, !3}

!0 = !{i32 2, i32 0}
!1 = !{!"AOMP_STANDALONE_11.6-2 clang version 11.0.0 (https://github.com/ROCm-Developer-Tools/amd-llvm-project 580cdea5fefa9c6e5175cb4143a21307cc43a9b3)"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"PIC Level", i32 1}
!4 = !{i32 9140, i32 9194, i32 9244, i32 9288}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"", !11, i64 0, !11, i64 8, !12, i64 16, !6, i64 24, !6, i64 32, !13, i64 40}
!11 = !{!"any pointer", !7, i64 0}
!12 = !{!"hsa_signal_s", !6, i64 0}
!13 = !{!"int", !7, i64 0}
!14 = !{!10, !13, i64 40}
!15 = !{!10, !11, i64 8}
!16 = !{!"exec"}
!17 = !{!18, !13, i64 16}
!18 = !{!"", !6, i64 0, !6, i64 8, !13, i64 16, !13, i64 20}
!19 = !{!18, !6, i64 8}
!20 = !{!18, !13, i64 20}
!21 = !{!18, !6, i64 0}

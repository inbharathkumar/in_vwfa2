July20 - "Gradation" Clustering

Pre-requisite: Read the report and the clustering code
If you've understood the report and the code, you would have noticed that we use some sort of local binning and scaling to overcome the anterior posterior issue. 

Why do I say that? [The following text is just my assumption - I am just a master's student and I am saying this purely based on the little knowledge I have, it's your duty to verify what I say is correct as per the current literature and scientific consensus] 

You see, typically, when we researchers use the mri_surfcluster algorithm to form clusters, it's true that they've got control over the cluster formation - like they can set some parameters like the t-value threshold, minimum size, etc.

In these scenarios, the researchers use some given functional localizer (as per their requirement) and they only consider the regions that have high activation for that stimuli. If a region has low activation, then they will simply reject it. 

However, in our case, we are trying to pickup the regions where the activation is weak. This is no big deal if it is done by a human expert. See the attached images of a brain plot without ROI, and with ROI outlines. 

If we ask the human expert to ignore the strength of the activation and simply draw outline over the clusters across the entire bigLOTS region, a human would be able to take a quick look at it and easily draw an outline. And if you look at the plot, the first obvious thing you will notice is that the ROI cluster in the anterior region is pale yellow and the ones in the posterior region is dark yellow. But that's not the point of focus here. 

What's so special in these human drawn outlines as opposed to the ones drawn by my BFS algorithm or the mri_surfcluster algorithm is that the human expert does not stick to a "HARD-CODED" threshold filter instead uses the "gradation" seen in the t-value activation i.e., when drawing the ROI outline in the posterior region, the human would pick a spot with dark red and draw an outline, but as the eye notices a stark decrease in the gradation, the human will mark that as the boundary, even if there are contiguous which are well above the threshold - similarly, in the anterior region, since we are now dealing with a low activation, the ROI outline includes vertices that are well below the threshold as long as they are contiguous and sticks close to the gradation, but the moment there's a decrease in the gradation, the human eye will capture it and the hand will automatically skip the areas outside that - you get what I am trying to say? I think you bots will call that bounding box or something? But I am not entirely sure if that term is used for fROI estimation. 

So, what do I want from you? If you notice, what I have done with my code implementation is something similar - but instead of making the algorithm to "SEE" the outlines clearly, I manipulated the activations to be brought out, so that the algorithm will be able to "SEE" the outlines and mark it for me. I don't think so this is a very elegant solution. 

#### Main Task: 
So, can you do a thorough check and tell me 

a) if there already exists a solution for my problem [IMPORTANT: A mandatory criteria for clustering algorithm is that the vertices have to be contiguous] BTW, I am bad at neuroscientific vocabulary, if there's any standard term that neuroscientists use to refer to this thing, tell me that.

and 

b) do you think we could brainstorm and come up with our own solution for the same? [IMPORTANT: A mandatory criteria for clustering algorithm is that the vertices have to be contiguous, apart from the peculiar "gradation" requirement] 

 If so, give me a simple python implementation of it, preferably the one that works with my existing code/dataset (a fail-proof sure-shot solution, not a made-up confabulated AI slop solution that will further waste my time and put me into permanent hell, if that's the case, you better say goodbye to me - that's better than pushing me into a hellhole) [IMPORTANT: A mandatory criteria for clustering algorithm is that the vertices have to be contiguous].

BTW, ignore the report writing task mentioned in the "01 - Context-Thesis Revival Workplan.txt" - that's for later once I am done with this. 

<side task>
[Assign separate agent for this task - provide it the entire context including the general instructions]
[Deliverable: Provide the final outputs of this task as files and make them available to be downloaded as a zip]

So, when I said this --> However, in our case, we are trying to pickup the regions where the activation is weak. 

But why is that the case? Because, VWFA is a tricky thing for fROI estimation. We already have several contrasts (you can take a look at the attachment for contrast information) including Checkerboard Pattern, Consonant String, False Font (to clearly filter out the activation of all processing other than the actual visual word recognition. Yet, we end up getting strong activation in the parietal region and low activation in the anterior region? 

Side Task: Take a thorough look at our project's methods (i.e., experimental setup and conditions) and then do a thorough literature review, and then tell me what could be the reason? 
1. Could it be because fMRI is so slow, and the activations in the anterior are so quick that it disappears before it gets captured unlike the activations in the posterior which stays for sometime for the fMRI to capture? [Remember, my supervisor (Garikoitz Lerma-Usabiaga) is an expert in neuroimaging including fMRI and dwi, if there was anything like that it would have been obvious for him and would have mentioned it in his paper. If not, he would have known when anyone in the field had published such a thing as their finding. So, if you say this would be the case, then I need strong reference, as well as the exact quote from that article]

2. Could it be because the cognitive demand required to do the process happening in that anterior region is less compared to the cognitive demand required to do the process occurring at posterior regions? 

3. Could it be because the processing happening in that region is subcortical and hence not elicited properly, whereas the ones happening in posterior is happening at cortical surface level and hence is shown with strong activation? 

4. Could it all be because of improper/not-so-well thought implmentation of GLM-I analysis? [I've attached the GLM-I analysis code as well, it's standard one, nothing fancy, probably suggested by an AI, regurgitating the standard template without taking into account the specifics of our project]

Please take my comments seriously and ensure you do thorough literature review for each one of these points carefully. If you forget and skip past after the first one, I will be in trouble because I will assume that you have looked into all of those before replying to me. So, better look into them in parallel so that I know for sure you have addressed all the points.

Apart from providing an answer to each of these doubts of mine, feel free to suggest me the correct reason [I mean, like I said, the reason might have been something obvious to you, but I am just a master's student and I want to let out my thoughts regardless of whether they sound silly or knowledgeable].
</end side task>

YouTube Video by Jamie Mitchell:
Title: Drawing ROIs in Freeview
Description: A quick how-to video showing how we draw regions of interest (ROIs) on the cortical surface using the Freeview viewer of FreeSurfer Suite. In this video, I show you how to load a cortical surface mesh, draw an anatomical ROI, project a statistical map to the surface, and use that statistical map to draw a function ROI. 
URL:  https://www.youtube.com/watch?v=HZ4Ec87zkIk
